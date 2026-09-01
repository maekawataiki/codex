mod auth;
mod auth_refresh;
mod catalog;
mod error;
mod mantle;
mod runtime;
mod runtime_catalog;

use std::path::PathBuf;
use std::sync::Arc;

use codex_api::ApiError;
use codex_api::Provider;
use codex_api::SharedAuthProvider;
use codex_api::TransportError;
use codex_login::AuthManager;
use codex_login::CodexAuth;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_5_6_LUNA_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_GPT_5_6_TERRA_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_RUNTIME_GLOBAL_GPT_5_6_LUNA_MODEL_ID;
use codex_model_provider_info::AMAZON_BEDROCK_RUNTIME_GLOBAL_GPT_5_6_TERRA_MODEL_ID;
use codex_model_provider_info::ModelProviderAwsAuthInfo;
use codex_model_provider_info::ModelProviderInfo;
use codex_models_manager::manager::SharedModelsManager;
use codex_models_manager::manager::StaticModelsManager;
use codex_protocol::account::ProviderAccount;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::openai_models::ModelsResponse;
use tokio::sync::OnceCell;

use crate::auth::auth_manager_for_provider;
use crate::auth::resolve_provider_auth as resolve_configured_provider_auth;
use crate::provider::ModelProvider;
use crate::provider::ModelProviderFuture;
use crate::provider::ProviderAccountResult;
use crate::provider::ProviderAccountState;
use crate::provider::ProviderAuthRecoveryMessages;
use crate::provider::ProviderCapabilities;
use crate::provider::ProviderUnauthorizedRecovery;
use crate::provider::RemoteCompactionSupport;
use crate::shared_state::process_shared_state;
use auth::auth_provider_from_method;
use auth::resolve_auth_method;
pub(crate) use auth_refresh::AwsAuthRecovery;
use catalog::normalize_bedrock_catalog;
pub(crate) use catalog::static_model_catalog;
use mantle::bedrock_mantle_runtime_base_url;
pub use mantle::is_supported_amazon_bedrock_region;
use runtime::bedrock_runtime_base_url;
use runtime_catalog::static_runtime_model_catalog;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BedrockEndpoint {
    Mantle,
    Runtime,
}

/// Runtime provider for Amazon Bedrock's OpenAI-compatible endpoints.
#[derive(Clone)]
pub(crate) struct AmazonBedrockModelProvider {
    pub(crate) info: ModelProviderInfo,
    aws: ModelProviderAwsAuthInfo,
    endpoint: BedrockEndpoint,
    auth_manager: Option<Arc<AuthManager>>,
    auth_recovery: Option<Arc<AwsAuthRecovery>>,
    /// Cached resolved base URL and auth provider. Populated on first call to
    /// `resolve_base_url_and_auth` so that `runtime_base_url()` and
    /// `api_auth()` share a single AWS SDK config load instead of each
    /// independently invoking `credential_process`.
    cached_auth: Arc<OnceCell<(Option<String>, SharedAuthProvider)>>,
}

impl std::fmt::Debug for AmazonBedrockModelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmazonBedrockModelProvider")
            .field("info", &self.info)
            .field("aws", &self.aws)
            .field("endpoint", &self.endpoint)
            .field("auth_manager", &self.auth_manager)
            .field("auth_recovery", &self.auth_recovery)
            .field("cached_auth", &self.cached_auth.initialized())
            .finish()
    }
}

impl AmazonBedrockModelProvider {
    pub(crate) fn new(
        provider_info: ModelProviderInfo,
        auth_manager: Option<Arc<AuthManager>>,
    ) -> Self {
        let endpoint = if provider_info.is_amazon_bedrock_runtime() {
            BedrockEndpoint::Runtime
        } else {
            BedrockEndpoint::Mantle
        };
        let aws = provider_info
            .aws
            .clone()
            .unwrap_or(ModelProviderAwsAuthInfo {
                profile: None,
                region: None,
                auth_refresh: None,
            });
        let uses_aws_sdk_auth = matches!(
            auth::auth_source(&provider_info, auth_manager.as_deref(), std::env::var),
            auth::BedrockAuthSource::ConfiguredAwsProfile | auth::BedrockAuthSource::AwsSdk
        );
        let auth_recovery = if uses_aws_sdk_auth && aws.auth_refresh.is_some() {
            process_shared_state().aws_auth_recovery(&aws)
        } else {
            None
        };
        let auth_manager = auth_manager_for_provider(auth_manager, &provider_info);
        Self {
            info: provider_info,
            aws,
            endpoint,
            auth_manager,
            auth_recovery,
            cached_auth: Arc::new(OnceCell::new()),
        }
    }

    fn auth_source(&self) -> auth::BedrockAuthSource {
        auth::auth_source(&self.info, self.auth_manager.as_deref(), std::env::var)
    }

    fn managed_auth(&self) -> Option<CodexAuth> {
        let source = self.auth_source();
        self.auth_manager
            .as_deref()
            .and_then(AuthManager::auth_cached)
            .filter(|auth| {
                matches!(
                    (source, auth),
                    (
                        auth::BedrockAuthSource::ManagedBearerToken,
                        CodexAuth::BedrockApiKey(_)
                    ) | (
                        auth::BedrockAuthSource::ManagedAccessKeys,
                        CodexAuth::BedrockAccessKeys(_)
                    )
                )
            })
    }

    fn uses_aws_auth_recovery(&self) -> bool {
        self.auth_recovery.is_some()
            && matches!(
                self.auth_source(),
                auth::BedrockAuthSource::ConfiguredAwsProfile | auth::BedrockAuthSource::AwsSdk
            )
    }

    async fn auth(&self) -> Option<CodexAuth> {
        match self.auth_source() {
            auth::BedrockAuthSource::CommandBearerToken => match self.auth_manager.as_ref() {
                Some(auth_manager) => auth_manager.auth().await,
                None => None,
            },
            auth::BedrockAuthSource::ManagedBearerToken
            | auth::BedrockAuthSource::ManagedAccessKeys => self.managed_auth(),
            auth::BedrockAuthSource::ConfiguredAwsProfile
            | auth::BedrockAuthSource::EnvBearerToken
            | auth::BedrockAuthSource::EnvAwsCredentials
            | auth::BedrockAuthSource::AwsSdk => None,
        }
    }

    async fn api_provider(&self) -> Result<Provider> {
        let mut api_provider_info = self.info.clone();
        api_provider_info.base_url = self.runtime_base_url().await?;
        api_provider_info.to_api_provider(/*auth_mode*/ None)
    }

    /// Resolves the Bedrock auth method once and returns both the base URL and
    /// the auth provider derived from that single resolution. This avoids
    /// loading the AWS SDK config (and invoking `credential_process`) multiple
    /// times per request cycle.
    async fn resolve_base_url_and_auth(&self) -> Result<(Option<String>, SharedAuthProvider)> {
        let source = self.auth_source();

        // Command bearer token uses a separate auth path that does not go
        // through `resolve_auth_method`, so handle it without the combined
        // resolution.
        if source == auth::BedrockAuthSource::CommandBearerToken {
            let base_url = self.runtime_base_url_only().await?;
            let auth = self.auth().await;
            let provider = resolve_configured_provider_auth(auth.as_ref(), &self.info)?;
            return Ok((base_url, provider));
        }

        // For all other auth sources, resolve the auth method exactly once
        // and derive both the base URL (from the region) and the auth
        // provider from the same result.
        let managed_auth = self.managed_auth();
        let method =
            resolve_auth_method(source, managed_auth.as_ref(), &self.aws, self.endpoint).await?;

        let base_url = if self.info.base_url.is_some() {
            self.info.base_url.clone()
        } else {
            let region = method.region();
            let url = match self.endpoint {
                BedrockEndpoint::Mantle => mantle::base_url(region)?,
                BedrockEndpoint::Runtime => runtime::base_url(region),
            };
            Some(url)
        };

        let auth_provider = auth_provider_from_method(method, self.endpoint);
        Ok((base_url, auth_provider))
    }

    /// Returns the base URL without resolving auth — used only for the command
    /// bearer token path where auth is resolved separately.
    async fn runtime_base_url_only(&self) -> Result<Option<String>> {
        if let Some(base_url) = self.info.base_url.clone() {
            return Ok(Some(base_url));
        }
        let auth_source = self.auth_source();
        let managed_auth = self.managed_auth();
        let base_url = match self.endpoint {
            BedrockEndpoint::Mantle => {
                bedrock_mantle_runtime_base_url(auth_source, managed_auth.as_ref(), &self.aws)
                    .await?
            }
            BedrockEndpoint::Runtime => {
                bedrock_runtime_base_url(auth_source, managed_auth.as_ref(), &self.aws).await?
            }
        };
        Ok(Some(base_url))
    }

    async fn runtime_base_url(&self) -> Result<Option<String>> {
        let (base_url, _) = self.get_or_resolve_auth().await?;
        Ok(base_url)
    }

    async fn api_auth(&self) -> Result<SharedAuthProvider> {
        let (_, auth_provider) = self.get_or_resolve_auth().await?;
        Ok(auth_provider)
    }

    /// Returns the cached `(base_url, auth_provider)` pair, resolving it on
    /// first call. This ensures that `credential_process` (and the full AWS
    /// SDK config load) is invoked at most once per provider instance rather
    /// than separately for each of `runtime_base_url()` and `api_auth()`.
    async fn get_or_resolve_auth(&self) -> Result<(Option<String>, SharedAuthProvider)> {
        self.cached_auth
            .get_or_try_init(|| self.resolve_base_url_and_auth())
            .await
            .cloned()
    }

    fn default_model_catalog(&self) -> ModelsResponse {
        match self.endpoint {
            BedrockEndpoint::Mantle => static_model_catalog(),
            BedrockEndpoint::Runtime => static_runtime_model_catalog(),
        }
    }
}

impl ModelProvider for AmazonBedrockModelProvider {
    fn info(&self) -> &ModelProviderInfo {
        &self.info
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            namespace_tools: true,
            image_generation: false,
            web_search: self.endpoint == BedrockEndpoint::Mantle,
            external_web_access: false,
            remote_compaction: RemoteCompactionSupport::V2,
        }
    }

    fn approval_review_preferred_model(&self) -> &'static str {
        match self.endpoint {
            BedrockEndpoint::Mantle => AMAZON_BEDROCK_GPT_5_6_LUNA_MODEL_ID,
            BedrockEndpoint::Runtime => AMAZON_BEDROCK_RUNTIME_GLOBAL_GPT_5_6_LUNA_MODEL_ID,
        }
    }

    fn memory_extraction_preferred_model(&self) -> &'static str {
        match self.endpoint {
            BedrockEndpoint::Mantle => AMAZON_BEDROCK_GPT_5_6_LUNA_MODEL_ID,
            BedrockEndpoint::Runtime => AMAZON_BEDROCK_RUNTIME_GLOBAL_GPT_5_6_LUNA_MODEL_ID,
        }
    }

    fn memory_consolidation_preferred_model(&self) -> &'static str {
        match self.endpoint {
            BedrockEndpoint::Mantle => AMAZON_BEDROCK_GPT_5_6_TERRA_MODEL_ID,
            BedrockEndpoint::Runtime => AMAZON_BEDROCK_RUNTIME_GLOBAL_GPT_5_6_TERRA_MODEL_ID,
        }
    }

    fn auth_manager(&self) -> Option<Arc<AuthManager>> {
        match self.auth_source() {
            auth::BedrockAuthSource::CommandBearerToken
            | auth::BedrockAuthSource::ManagedBearerToken
            | auth::BedrockAuthSource::ManagedAccessKeys => self.auth_manager.clone(),
            auth::BedrockAuthSource::ConfiguredAwsProfile
            | auth::BedrockAuthSource::EnvBearerToken
            | auth::BedrockAuthSource::EnvAwsCredentials
            | auth::BedrockAuthSource::AwsSdk => None,
        }
    }

    fn is_recoverable_auth_error(&self, error: &TransportError) -> bool {
        matches!(
            error,
            TransportError::Http { status, .. } if *status == http::StatusCode::UNAUTHORIZED
        ) || (self.uses_aws_auth_recovery() && error::is_refreshable_auth_error(error))
    }

    fn auth_recovery_messages(&self) -> Option<ProviderAuthRecoveryMessages> {
        self.uses_aws_auth_recovery()
            .then_some(ProviderAuthRecoveryMessages {
                started: "AWS session has expired. Reauthenticating...",
                succeeded: "Signed in with AWS.",
            })
    }

    fn recover_from_unauthorized(
        &self,
    ) -> ModelProviderFuture<'_, Result<ProviderUnauthorizedRecovery>> {
        Box::pin(async move {
            let Some(recovery) = self
                .auth_recovery
                .as_ref()
                .filter(|_| self.uses_aws_auth_recovery())
            else {
                return Ok(ProviderUnauthorizedRecovery::NotConfigured);
            };

            recovery.refresh().await.map_err(|error| {
                if error.kind() == std::io::ErrorKind::InvalidInput {
                    CodexErr::InvalidRequest(error.to_string())
                } else {
                    CodexErr::Io(error)
                }
            })?;
            Ok(ProviderUnauthorizedRecovery::Recovered)
        })
    }

    fn auth(&self) -> ModelProviderFuture<'_, Option<CodexAuth>> {
        Box::pin(AmazonBedrockModelProvider::auth(self))
    }

    fn account_state(&self) -> ProviderAccountResult {
        Ok(ProviderAccountState {
            account: Some(ProviderAccount::AmazonBedrock {
                uses_codex_managed_credentials: matches!(
                    self.auth_source(),
                    auth::BedrockAuthSource::ManagedBearerToken
                        | auth::BedrockAuthSource::ManagedAccessKeys
                ),
            }),
            requires_openai_auth: false,
        })
    }

    fn map_api_error(&self, error: ApiError) -> CodexErr {
        error::map_api_error(error)
    }

    fn api_provider(&self) -> ModelProviderFuture<'_, Result<Provider>> {
        Box::pin(AmazonBedrockModelProvider::api_provider(self))
    }

    fn runtime_base_url(&self) -> ModelProviderFuture<'_, Result<Option<String>>> {
        Box::pin(AmazonBedrockModelProvider::runtime_base_url(self))
    }

    fn api_auth(&self) -> ModelProviderFuture<'_, Result<SharedAuthProvider>> {
        Box::pin(AmazonBedrockModelProvider::api_auth(self))
    }

    fn models_manager(
        &self,
        _codex_home: PathBuf,
        config_model_catalog: Option<ModelsResponse>,
    ) -> SharedModelsManager {
        Arc::new(StaticModelsManager::new(
            /*auth_manager*/ None,
            config_model_catalog
                .map_or_else(|| self.default_model_catalog(), normalize_bedrock_catalog),
        ))
    }

    fn models_manager_without_cache(
        &self,
        config_model_catalog: Option<ModelsResponse>,
    ) -> SharedModelsManager {
        Arc::new(StaticModelsManager::new(
            /*auth_manager*/ None,
            config_model_catalog
                .map_or_else(|| self.default_model_catalog(), normalize_bedrock_catalog),
        ))
    }
}

#[cfg(test)]
#[path = "error_tests.rs"]
mod error_tests;

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use codex_login::auth::BedrockAccessKeysAuth;
    use codex_login::auth::BedrockApiKeyAuth;
    use codex_model_provider_info::AwsAuthRefreshConfig;
    use codex_protocol::config_types::ModelProviderAuthInfo;
    use http::HeaderValue;
    use pretty_assertions::assert_eq;

    use super::*;

    fn command_auth_provider(base_url: Option<&str>) -> ModelProviderInfo {
        let mut provider = ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None);
        provider.base_url = base_url.map(str::to_string);
        provider.auth = Some(ModelProviderAuthInfo {
            command: "token-fetcher".to_string(),
            args: vec!["fetch".into()],
            timeout_ms: NonZeroU64::new(5_000).expect("timeout should be non-zero"),
            refresh_interval_ms: 300_000,
            cwd: std::env::current_dir()
                .expect("current directory should be available")
                .try_into()
                .expect("current directory should be absolute"),
        });
        provider
    }

    #[test]
    fn api_provider_for_bedrock_bearer_token_uses_configured_region_endpoint() {
        let region = "eu-central-1";
        let mut api_provider_info =
            ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None);
        api_provider_info.base_url = Some(mantle::base_url(region).expect("supported region"));
        let api_provider = api_provider_info
            .to_api_provider(/*auth_mode*/ None)
            .expect("api provider should build");

        assert_eq!(
            api_provider.base_url,
            "https://bedrock-mantle.eu-central-1.api.aws/openai/v1"
        );
    }

    #[tokio::test]
    async fn command_auth_resolves_configured_and_regional_base_urls() {
        let mut provider_info = command_auth_provider(Some("https://proxy.example.com/v1"));
        provider_info.aws = Some(ModelProviderAwsAuthInfo {
            profile: Some("aws-profile-that-should-not-be-loaded".to_string()),
            region: Some("us-west-2".to_string()),
            auth_refresh: Some(AwsAuthRefreshConfig {
                command: "aws".to_string(),
                args: vec!["login".into()],
                timeout_ms: NonZeroU64::new(1_000).expect("timeout should be non-zero"),
            }),
        });
        let provider = AmazonBedrockModelProvider::new(provider_info, /*auth_manager*/ None);

        assert_eq!(
            provider
                .runtime_base_url()
                .await
                .expect("configured base URL should resolve"),
            Some("https://proxy.example.com/v1".to_string())
        );
        assert!(
            provider
                .auth_manager()
                .expect("command auth manager should be exposed")
                .has_external_auth()
        );
        assert_eq!(
            provider.account_state(),
            Ok(ProviderAccountState {
                account: Some(ProviderAccount::AmazonBedrock {
                    uses_codex_managed_credentials: false,
                }),
                requires_openai_auth: false,
            })
        );

        let mut regional_provider_info = command_auth_provider(/*base_url*/ None);
        regional_provider_info.aws = Some(ModelProviderAwsAuthInfo {
            profile: None,
            region: Some("us-west-2".to_string()),
            auth_refresh: None,
        });
        let regional_provider =
            AmazonBedrockModelProvider::new(regional_provider_info, /*auth_manager*/ None);

        assert_eq!(
            regional_provider
                .runtime_base_url()
                .await
                .expect("configured AWS region should resolve for command auth"),
            Some("https://bedrock-mantle.us-west-2.api.aws/openai/v1".to_string())
        );
    }

    #[tokio::test]
    async fn configured_profile_takes_precedence_over_managed_auth() {
        let managed_auth = BedrockApiKeyAuth {
            api_key: "managed-bedrock-api-key".to_string(),
            region: "us-east-1".to_string(),
        };
        let auth_manager =
            AuthManager::from_auth_for_testing(CodexAuth::BedrockApiKey(managed_auth.clone()));
        let aws = ModelProviderAwsAuthInfo {
            profile: None,
            region: Some("us-west-2".to_string()),
            auth_refresh: Some(AwsAuthRefreshConfig {
                command: "aws".to_string(),
                args: vec!["login".into()],
                timeout_ms: NonZeroU64::new(1_000).expect("timeout should be non-zero"),
            }),
        };
        let provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_provider(Some(aws.clone())),
            Some(auth_manager.clone()),
        );

        assert!(Arc::ptr_eq(
            &provider
                .auth_manager()
                .expect("managed Bedrock auth manager should be exposed"),
            &auth_manager,
        ));
        assert_eq!(
            provider.auth().await,
            Some(CodexAuth::BedrockApiKey(managed_auth))
        );
        assert_eq!(
            provider.account_state(),
            Ok(ProviderAccountState {
                account: Some(ProviderAccount::AmazonBedrock {
                    uses_codex_managed_credentials: true,
                }),
                requires_openai_auth: false,
            })
        );
        assert_eq!(
            provider
                .runtime_base_url()
                .await
                .expect("managed Bedrock region should resolve"),
            Some("https://bedrock-mantle.us-east-1.api.aws/openai/v1".to_string())
        );
        assert_eq!(
            provider
                .api_auth()
                .await
                .expect("managed Bedrock auth should resolve")
                .to_auth_headers()
                .get(http::header::AUTHORIZATION),
            Some(&HeaderValue::from_static("Bearer managed-bedrock-api-key"))
        );
        assert!(!provider.uses_aws_auth_recovery());

        let access_keys_auth_manager = AuthManager::from_auth_for_testing(
            CodexAuth::BedrockAccessKeys(BedrockAccessKeysAuth {
                access_key_id: "managed-access-key-id".to_string(),
                secret_access_key: "managed-secret-access-key".to_string(),
                session_token: None,
            }),
        );

        for auth_manager in [auth_manager, access_keys_auth_manager] {
            let configured_profile_provider = AmazonBedrockModelProvider::new(
                ModelProviderInfo::create_amazon_bedrock_provider(Some(ModelProviderAwsAuthInfo {
                    profile: Some("configured-aws-profile".to_string()),
                    ..aws.clone()
                })),
                Some(auth_manager),
            );

            assert!(configured_profile_provider.auth_manager().is_none());
            assert_eq!(configured_profile_provider.auth().await, None);
            assert_eq!(
                configured_profile_provider.account_state(),
                Ok(ProviderAccountState {
                    account: Some(ProviderAccount::AmazonBedrock {
                        uses_codex_managed_credentials: false,
                    }),
                    requires_openai_auth: false,
                })
            );
            assert_eq!(
                configured_profile_provider
                    .runtime_base_url()
                    .await
                    .expect("configured AWS profile region should resolve"),
                Some("https://bedrock-mantle.us-west-2.api.aws/openai/v1".to_string())
            );
            assert!(
                configured_profile_provider
                    .api_auth()
                    .await
                    .expect("configured AWS profile auth should resolve")
                    .to_auth_headers()
                    .get(http::header::AUTHORIZATION)
                    .is_none()
            );
            assert!(configured_profile_provider.uses_aws_auth_recovery());
        }
    }

    #[tokio::test]
    async fn openai_auth_is_not_exposed_to_bedrock() {
        let provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None),
            Some(AuthManager::from_auth_for_testing(CodexAuth::from_api_key(
                "openai-api-key",
            ))),
        );

        assert!(provider.auth_manager().is_none());
        assert_eq!(provider.auth().await, None);
        assert_eq!(
            provider.account_state(),
            Ok(ProviderAccountState {
                account: Some(ProviderAccount::AmazonBedrock {
                    uses_codex_managed_credentials: false,
                }),
                requires_openai_auth: false,
            })
        );
    }

    #[test]
    fn capabilities_enable_web_search_but_disable_image_generation() {
        let provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None),
            /*auth_manager*/ None,
        );

        assert_eq!(
            provider.capabilities(),
            ProviderCapabilities {
                namespace_tools: true,
                image_generation: false,
                web_search: true,
                external_web_access: false,
                remote_compaction: RemoteCompactionSupport::V2,
            }
        );
    }

    #[test]
    fn runtime_capabilities_disable_web_search_and_support_v2_remote_compaction() {
        let provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_runtime_provider(/*aws*/ None),
            /*auth_manager*/ None,
        );

        assert_eq!(
            provider.capabilities(),
            ProviderCapabilities {
                namespace_tools: true,
                image_generation: false,
                web_search: false,
                external_web_access: false,
                remote_compaction: RemoteCompactionSupport::V2,
            }
        );
    }

    #[tokio::test]
    async fn runtime_managed_auth_resolves_runtime_endpoint() {
        let managed_auth = BedrockApiKeyAuth {
            api_key: "managed-bedrock-api-key".to_string(),
            region: "eu-west-1".to_string(),
        };
        let auth_manager =
            AuthManager::from_auth_for_testing(CodexAuth::BedrockApiKey(managed_auth));
        let provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_runtime_provider(/*aws*/ None),
            Some(auth_manager),
        );

        assert_eq!(
            provider
                .runtime_base_url()
                .await
                .expect("managed Bedrock Runtime region should resolve"),
            Some("https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1".to_string())
        );
    }

    #[test]
    fn preferred_background_models_match_bedrock_endpoint() {
        let mantle_provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_provider(/*aws*/ None),
            /*auth_manager*/ None,
        );
        let runtime_provider = AmazonBedrockModelProvider::new(
            ModelProviderInfo::create_amazon_bedrock_runtime_provider(/*aws*/ None),
            /*auth_manager*/ None,
        );

        assert_eq!(
            (
                mantle_provider.approval_review_preferred_model(),
                mantle_provider.memory_extraction_preferred_model(),
                mantle_provider.memory_consolidation_preferred_model(),
            ),
            (
                AMAZON_BEDROCK_GPT_5_6_LUNA_MODEL_ID,
                AMAZON_BEDROCK_GPT_5_6_LUNA_MODEL_ID,
                AMAZON_BEDROCK_GPT_5_6_TERRA_MODEL_ID,
            )
        );
        assert_eq!(
            (
                runtime_provider.approval_review_preferred_model(),
                runtime_provider.memory_extraction_preferred_model(),
                runtime_provider.memory_consolidation_preferred_model(),
            ),
            (
                AMAZON_BEDROCK_RUNTIME_GLOBAL_GPT_5_6_LUNA_MODEL_ID,
                AMAZON_BEDROCK_RUNTIME_GLOBAL_GPT_5_6_LUNA_MODEL_ID,
                AMAZON_BEDROCK_RUNTIME_GLOBAL_GPT_5_6_TERRA_MODEL_ID,
            )
        );
    }
}
