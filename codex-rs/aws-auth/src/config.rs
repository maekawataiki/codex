use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;

use aws_config::BehaviorVersion;
use aws_config::SdkConfig;
use aws_credential_types::provider::SharedCredentialsProvider;
use aws_types::region::Region;
use tokio::sync::Mutex;

use crate::AwsAuthConfig;
use crate::AwsAuthError;

/// Cache key derived from the AWS auth config fields that affect SDK config
/// loading (profile name and region). Different services using the same
/// profile+region can share an `SdkConfig` because the credentials and region
/// are identical — only the service name used for SigV4 signing differs, and
/// that is not part of the `SdkConfig`.
type SdkConfigCacheKey = (Option<String>, Option<String>);

/// Process-wide cache of loaded `SdkConfig` instances. This avoids invoking
/// `credential_process` (and the full provider chain walk) multiple times for
/// the same profile+region combination within a single process.
static SDK_CONFIG_CACHE: OnceLock<Mutex<HashMap<SdkConfigCacheKey, Arc<SdkConfig>>>> =
    OnceLock::new();

fn sdk_config_cache() -> &'static Mutex<HashMap<SdkConfigCacheKey, Arc<SdkConfig>>> {
    SDK_CONFIG_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub(crate) async fn load_sdk_config(
    config: &AwsAuthConfig,
) -> Result<Arc<SdkConfig>, AwsAuthError> {
    if config.service.trim().is_empty() {
        return Err(AwsAuthError::EmptyService);
    }

    let cache_key: SdkConfigCacheKey = (config.profile.clone(), config.region.clone());

    // Fast path: return cached config if available.
    {
        let cache = sdk_config_cache().lock().await;
        if let Some(cached) = cache.get(&cache_key) {
            return Ok(Arc::clone(cached));
        }
    }

    // Slow path: load from the SDK (may invoke credential_process).
    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(profile) = config.profile.as_ref() {
        loader = loader.profile_name(profile);
    }
    if let Some(region) = config.region.as_ref() {
        loader = loader.region(Region::new(region.clone()));
    }

    let sdk_config = Arc::new(loader.load().await);

    // Store in cache for subsequent calls.
    {
        let mut cache = sdk_config_cache().lock().await;
        cache.insert(cache_key, Arc::clone(&sdk_config));
    }

    Ok(sdk_config)
}

pub(crate) fn credentials_provider(
    sdk_config: &SdkConfig,
) -> Result<SharedCredentialsProvider, AwsAuthError> {
    sdk_config
        .credentials_provider()
        .ok_or(AwsAuthError::MissingCredentialsProvider)
}

pub(crate) fn resolved_region(sdk_config: &SdkConfig) -> Result<String, AwsAuthError> {
    sdk_config
        .region()
        .map(ToString::to_string)
        .ok_or(AwsAuthError::MissingRegion)
}
