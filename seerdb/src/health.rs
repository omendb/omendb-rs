// Health check system for production monitoring
// Detects degraded performance and critical conditions

use std::fmt;

/// Overall health status of the database
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthStatus {
    /// Overall health: true if all checks are healthy
    pub healthy: bool,
    /// Individual health checks
    pub checks: Vec<HealthCheck>,
}

impl HealthStatus {
    /// Create a new health status
    pub fn new(checks: Vec<HealthCheck>) -> Self {
        let healthy = checks.iter().all(|c| c.status == CheckStatus::Healthy);
        Self { healthy, checks }
    }

    /// Check if any check is unhealthy
    pub fn has_unhealthy(&self) -> bool {
        self.checks
            .iter()
            .any(|c| c.status == CheckStatus::Unhealthy)
    }

    /// Check if any check is degraded
    pub fn has_degraded(&self) -> bool {
        self.checks
            .iter()
            .any(|c| c.status == CheckStatus::Degraded)
    }
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.healthy {
            writeln!(f, "✅ Database is HEALTHY")?;
        } else if self.has_unhealthy() {
            writeln!(f, "❌ Database is UNHEALTHY")?;
        } else {
            writeln!(f, "⚠️  Database is DEGRADED")?;
        }

        for check in &self.checks {
            writeln!(f, "  {} {}", check.status_icon(), check)?;
        }

        Ok(())
    }
}

/// Individual health check result
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthCheck {
    /// Name of the check (e.g., "disk_space", "compaction_lag")
    pub name: String,
    /// Status of this check
    pub status: CheckStatus,
    /// Optional message with details
    pub message: Option<String>,
}

impl HealthCheck {
    /// Create a healthy check
    pub fn healthy(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Healthy,
            message: None,
        }
    }

    /// Create a healthy check with a message
    pub fn healthy_with_message(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Healthy,
            message: Some(message.into()),
        }
    }

    /// Create a degraded check
    pub fn degraded(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Degraded,
            message: Some(message.into()),
        }
    }

    /// Create an unhealthy check
    pub fn unhealthy(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: CheckStatus::Unhealthy,
            message: Some(message.into()),
        }
    }

    fn status_icon(&self) -> &'static str {
        match self.status {
            CheckStatus::Healthy => "✅",
            CheckStatus::Degraded => "⚠️ ",
            CheckStatus::Unhealthy => "❌",
        }
    }
}

impl fmt::Display for HealthCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.name, self.status)?;
        if let Some(ref msg) = self.message {
            write!(f, " - {}", msg)?;
        }
        Ok(())
    }
}

/// Status of a health check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckStatus {
    /// Check passed, system is operating normally
    Healthy,
    /// Check detected degraded performance, but system is still functional
    Degraded,
    /// Check failed, system is in critical condition
    Unhealthy,
}

impl fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckStatus::Healthy => write!(f, "HEALTHY"),
            CheckStatus::Degraded => write!(f, "DEGRADED"),
            CheckStatus::Unhealthy => write!(f, "UNHEALTHY"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_all_healthy() {
        let checks = vec![
            HealthCheck::healthy("disk_space"),
            HealthCheck::healthy("compaction_lag"),
            HealthCheck::healthy("wal_size"),
        ];

        let status = HealthStatus::new(checks);
        assert!(status.healthy);
        assert!(!status.has_degraded());
        assert!(!status.has_unhealthy());
    }

    #[test]
    fn test_health_status_degraded() {
        let checks = vec![
            HealthCheck::healthy("disk_space"),
            HealthCheck::degraded("compaction_lag", "L0 has 12 SSTables (>10)"),
            HealthCheck::healthy("wal_size"),
        ];

        let status = HealthStatus::new(checks);
        assert!(!status.healthy);
        assert!(status.has_degraded());
        assert!(!status.has_unhealthy());
    }

    #[test]
    fn test_health_status_unhealthy() {
        let checks = vec![
            HealthCheck::healthy("disk_space"),
            HealthCheck::unhealthy("compaction_lag", "L0 has 25 SSTables (>20)"),
            HealthCheck::healthy("wal_size"),
        ];

        let status = HealthStatus::new(checks);
        assert!(!status.healthy);
        assert!(!status.has_degraded());
        assert!(status.has_unhealthy());
    }

    #[test]
    fn test_health_check_display() {
        let check = HealthCheck::degraded("compaction_lag", "L0 has 12 SSTables");
        let display = format!("{}", check);
        assert!(display.contains("compaction_lag"));
        assert!(display.contains("DEGRADED"));
        assert!(display.contains("L0 has 12 SSTables"));
    }

    #[test]
    fn test_health_status_display() {
        let checks = vec![
            HealthCheck::healthy("disk_space"),
            HealthCheck::degraded("compaction_lag", "L0 has 12 SSTables"),
        ];

        let status = HealthStatus::new(checks);
        let display = format!("{}", status);
        assert!(display.contains("DEGRADED"));
        assert!(display.contains("disk_space"));
        assert!(display.contains("compaction_lag"));
    }
}
