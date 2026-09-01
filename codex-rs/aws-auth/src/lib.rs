mod config;
mod discovery;
mod signing;

use std::sync::Arc;
use std::time::SystemTime;

use aws_credential_types::Credentials;
use aws_credential_types::provider::ProvideCredentials;
use aws_credential_types::provider::SharedCredentialsProvider;
use bytes::Bytes;
use http::HeaderMap;
use http::Method;
use thiserror::Error;
use tokio::sync::Mutex;

pub use discovery::AwsProfile;
pub use discovery::discover_aws_profiles;
pub use discovery::validate_aws_profile;

/// AWS auth configuration used to resolve credentials and sign requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwsAuthConfig {
    pub profile: Option<String>,
    pub region: Option<String>,
    pub service: String,
}

/// Static AWS access keys supplied by a caller instead of the default SDK chain.
#[derive(Clone, PartialEq, Eq)]
pub struct AwsAccessKeys {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

impl std::fmt::Debug for AwsAccessKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AwsAccessKeys")
            .field("access_key_id", &"<redacted>")
            .field("secret_access_key", &"<redacted>")
            .field(
                "session_token",
                &self.session_token.as_ref().map(|_| "<redacted>"),
            )
            .finish()
    }
}

/// Generic HTTP request shape consumed by SigV4 signing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwsRequestToSign {
    pub method: Method,
    pub url: String,
    pub headers: HeaderMap,
    pub body: Bytes,
}

/// Signed request parts returned to the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AwsSignedRequest {
    pub url: String,
    pub headers: HeaderMap,
}

/// Errors returned by credential loading or SigV4 signing.
#[derive(Debug, Error)]
pub enum AwsAuthError {
    #[error("AWS service name must not be empty")]
    EmptyService,
    #[error("AWS profile must be configured")]
    MissingProfile,
    #[error("AWS SDK config did not resolve a credentials provider")]
    MissingCredentialsProvider,
    #[error("AWS SDK config did not resolve a region")]
    MissingRegion,
    #[error("failed to load AWS profiles: {0}")]
    ProfileLoad(#[from] aws_config::profile::ProfileFileLoadError),
    #[error("failed to load AWS credentials: {0}")]
    Credentials(#[from] aws_credential_types::provider::error::CredentialsError),
    #[error("request URL is not a valid URI: {0}")]
    InvalidUri(#[source] http::uri::InvalidUri),
    #[error("failed to construct HTTP request for signing: {0}")]
    BuildHttpRequest(#[source] http::Error),
    #[error("request contains a non-UTF8 header value: {0}")]
    InvalidHeaderValue(#[source] http::header::ToStrError),
    #[error("failed to build signable request: {0}")]
    SigningRequest(#[source] aws_sigv4::http_request::SigningError),
    #[error("failed to build SigV4 signing params: {0}")]
    SigningParams(String),
    #[error("SigV4 signing failed: {0}")]
    SigningFailure(#[source] aws_sigv4::http_request::SigningError),
}

/// Loaded AWS auth context that can sign outbound HTTP requests.
#[derive(Clone)]
pub struct AwsAuthContext {
    credentials_provider: SharedCredentialsProvider,
    /// Cached credentials from the last successful `provide_credentials()` call.
    /// This avoids re-invoking `credential_process` on every SigV4 signing
    /// operation within the same session.
    cached_credentials: Arc<Mutex<Option<Credentials>>>,
    region: String,
    service: String,
}

impl std::fmt::Debug for AwsAuthContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AwsAuthContext")
            .field("region", &self.region)
            .field("service", &self.service)
            .finish_non_exhaustive()
    }
}

impl AwsAuthContext {
    pub async fn load(config: AwsAuthConfig) -> Result<Self, AwsAuthError> {
        let sdk_config = config::load_sdk_config(&config).await?;
        let credentials_provider = config::credentials_provider(&sdk_config)?;
        let region = config::resolved_region(&sdk_config)?;

        Ok(Self {
            credentials_provider,
            cached_credentials: Arc::new(Mutex::new(None)),
            region,
            service: config.service.trim().to_string(),
        })
    }

    pub async fn load_with_access_keys(
        config: AwsAuthConfig,
        access_keys: AwsAccessKeys,
    ) -> Result<Self, AwsAuthError> {
        let mut context = Self::load(config).await?;
        context.credentials_provider =
            SharedCredentialsProvider::new(aws_credential_types::Credentials::new(
                access_keys.access_key_id,
                access_keys.secret_access_key,
                access_keys.session_token,
                /*expires_after*/ None,
                "codex-managed-bedrock-access-keys",
            ));
        Ok(context)
    }

    pub async fn load_profile(config: AwsAuthConfig) -> Result<Self, AwsAuthError> {
        let profile = config
            .profile
            .as_deref()
            .ok_or(AwsAuthError::MissingProfile)?;
        let credentials_provider = SharedCredentialsProvider::new(
            discovery::profile_credentials_provider(profile, config.region.as_deref()).await,
        );
        let mut context = Self::load(config).await?;
        context.credentials_provider = credentials_provider;
        Ok(context)
    }

    pub fn region(&self) -> &str {
        &self.region
    }

    pub fn service(&self) -> &str {
        &self.service
    }

    pub async fn sign(&self, request: AwsRequestToSign) -> Result<AwsSignedRequest, AwsAuthError> {
        self.sign_at(request, SystemTime::now()).await
    }

    async fn sign_at(
        &self,
        request: AwsRequestToSign,
        time: SystemTime,
    ) -> Result<AwsSignedRequest, AwsAuthError> {
        let credentials = self.get_or_refresh_credentials().await?;
        signing::sign_request(&credentials, &self.region, &self.service, request, time)
    }

    /// Returns cached credentials if they are still valid (more than 5 minutes
    /// until expiry), otherwise fetches fresh ones from the provider and caches
    /// them.
    async fn get_or_refresh_credentials(&self) -> Result<Credentials, AwsAuthError> {
        // Check cache without holding the lock across an await point.
        {
            let cached = self.cached_credentials.lock().await;
            if let Some(creds) = cached.as_ref().filter(|c| credentials_still_valid(c)) {
                return Ok(creds.clone());
            }
        }

        // Cache miss or expired — fetch fresh credentials. This await is
        // outside the lock so concurrent callers can still read cached values.
        let fresh = self.credentials_provider.provide_credentials().await?;

        {
            let mut cached = self.cached_credentials.lock().await;
            *cached = Some(fresh.clone());
        }

        Ok(fresh)
    }
}

/// Returns `true` if the credentials are still usable (more than 5 minutes
/// until expiry, or no expiry at all for static credentials).
fn credentials_still_valid(creds: &Credentials) -> bool {
    match creds.expiry() {
        Some(expiry) => {
            let remaining = expiry.duration_since(SystemTime::now()).unwrap_or_default();
            remaining.as_secs() > 300
        }
        // No expiry = static credentials, always valid.
        None => true,
    }
}

impl AwsAuthError {
    /// Returns whether retrying the outbound request can reasonably recover from this auth error.
    pub fn is_retryable(&self) -> bool {
        match self {
            AwsAuthError::Credentials(error) => matches!(
                error,
                aws_credential_types::provider::error::CredentialsError::ProviderTimedOut(_)
                    | aws_credential_types::provider::error::CredentialsError::ProviderError(_)
            ),
            AwsAuthError::EmptyService
            | AwsAuthError::MissingProfile
            | AwsAuthError::MissingCredentialsProvider
            | AwsAuthError::MissingRegion
            | AwsAuthError::ProfileLoad(_)
            | AwsAuthError::InvalidUri(_)
            | AwsAuthError::BuildHttpRequest(_)
            | AwsAuthError::InvalidHeaderValue(_)
            | AwsAuthError::SigningRequest(_)
            | AwsAuthError::SigningParams(_)
            | AwsAuthError::SigningFailure(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;
    use std::time::UNIX_EPOCH;

    use aws_credential_types::Credentials;
    use aws_credential_types::provider::error::CredentialsError;
    use pretty_assertions::assert_eq;

    use super::*;

    fn test_context(session_token: Option<&str>) -> AwsAuthContext {
        AwsAuthContext {
            credentials_provider: SharedCredentialsProvider::new(Credentials::new(
                "AKIDEXAMPLE",
                "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
                session_token.map(str::to_string),
                /*expires_after*/ None,
                "unit-test",
            )),
            cached_credentials: Arc::new(Mutex::new(None)),
            region: "us-east-1".to_string(),
            service: "bedrock".to_string(),
        }
    }

    fn test_request() -> AwsRequestToSign {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        headers.insert("x-test-header", http::HeaderValue::from_static("present"));
        AwsRequestToSign {
            method: Method::POST,
            url: "https://bedrock-runtime.us-east-1.amazonaws.com/v1/responses".to_string(),
            headers,
            body: Bytes::from_static(br#"{"model":"openai.gpt-oss-120b-1:0"}"#),
        }
    }

    #[tokio::test]
    async fn sign_adds_sigv4_headers_and_preserves_existing_headers() {
        let signed = test_context(/*session_token*/ None)
            .sign_at(
                test_request(),
                UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            )
            .await
            .expect("request should sign");

        assert_eq!(
            signing::header_value(&signed.headers, http::header::CONTENT_TYPE.as_str()),
            Some("application/json".to_string())
        );
        assert_eq!(
            signing::header_value(&signed.headers, "x-test-header"),
            Some("present".to_string())
        );
        assert_eq!(
            signed.url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/v1/responses"
        );
        assert!(
            signing::header_value(&signed.headers, http::header::AUTHORIZATION.as_str())
                .is_some_and(|value| value.starts_with("AWS4-HMAC-SHA256 "))
        );
        assert!(signing::header_value(&signed.headers, "x-amz-date").is_some());
    }

    #[test]
    fn credentials_provider_failures_are_retryable() {
        assert!(
            AwsAuthError::Credentials(CredentialsError::provider_error("temporarily unavailable"))
                .is_retryable()
        );
        assert!(
            AwsAuthError::Credentials(CredentialsError::provider_timed_out(Duration::from_secs(1)))
                .is_retryable()
        );
    }

    #[test]
    fn deterministic_aws_auth_errors_are_not_retryable() {
        assert!(!AwsAuthError::EmptyService.is_retryable());
        assert!(!AwsAuthError::MissingProfile.is_retryable());
        assert!(
            !AwsAuthError::Credentials(CredentialsError::not_loaded_no_source()).is_retryable()
        );
        assert!(
            !AwsAuthError::Credentials(CredentialsError::invalid_configuration("bad profile"))
                .is_retryable()
        );
        assert!(
            !AwsAuthError::Credentials(CredentialsError::unhandled("unexpected response"))
                .is_retryable()
        );
    }

    #[tokio::test]
    async fn sign_includes_session_token_when_credentials_have_one() {
        let signed = test_context(Some("session-token"))
            .sign_at(
                test_request(),
                UNIX_EPOCH + Duration::from_secs(1_700_000_000),
            )
            .await
            .expect("request should sign");

        assert_eq!(
            signing::header_value(&signed.headers, "x-amz-security-token"),
            Some("session-token".to_string())
        );
    }

    #[tokio::test]
    async fn load_rejects_invalid_configuration() {
        let err = AwsAuthContext::load(AwsAuthConfig {
            profile: None,
            region: None,
            service: "   ".to_string(),
        })
        .await
        .expect_err("empty service should be rejected");

        assert_eq!(err.to_string(), "AWS service name must not be empty");

        let err = AwsAuthContext::load_profile(AwsAuthConfig {
            profile: None,
            region: Some("us-east-1".to_string()),
            service: "bedrock".to_string(),
        })
        .await
        .expect_err("profile auth should require a configured profile");

        assert_eq!(err.to_string(), "AWS profile must be configured");
    }

    /// A credential provider that counts how many times it is invoked.
    #[derive(Debug)]
    struct CountingCredentialsProvider {
        call_count: Arc<std::sync::atomic::AtomicU32>,
    }

    impl aws_credential_types::provider::ProvideCredentials for CountingCredentialsProvider {
        fn provide_credentials<'a>(
            &'a self,
        ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            self.call_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            aws_credential_types::provider::future::ProvideCredentials::ready(Ok(Credentials::new(
                "AKIDEXAMPLE",
                "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
                None,
                Some(UNIX_EPOCH + Duration::from_secs(4_100_000_000)),
                "counting-test",
            )))
        }
    }

    #[tokio::test]
    async fn concurrent_signing_invokes_credential_provider_once() {
        let call_count = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let counter = call_count.clone();
        let provider = CountingCredentialsProvider { call_count };
        let context = AwsAuthContext {
            credentials_provider: SharedCredentialsProvider::new(provider),
            cached_credentials: Arc::new(Mutex::new(None)),
            region: "us-east-1".to_string(),
            service: "bedrock".to_string(),
        };

        let time = UNIX_EPOCH + Duration::from_secs(1_700_000_000);

        // Fire 10 concurrent sign() calls.
        let mut handles = Vec::new();
        for _ in 0..10 {
            let ctx = context.clone();
            handles.push(tokio::spawn(async move {
                ctx.sign_at(test_request(), time)
                    .await
                    .expect("signing should succeed")
            }));
        }

        for handle in handles {
            handle.await.expect("task should not panic");
        }

        // The credential provider should be called at most once because all
        // concurrent callers share the cached credentials.
        assert_eq!(
            counter.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "credential provider should be invoked exactly once across concurrent sign() calls"
        );
    }
}
