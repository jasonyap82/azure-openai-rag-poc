# Contoso Cloud Services - Security and Compliance

## Identity and access

All Contoso services support federated identity through SAML 2.0 and OpenID Connect.
Local accounts are disabled by default on Enterprise plans and cannot be re-enabled
without a signed exception request.

Multi-factor authentication is mandatory for all accounts holding the Owner or
Security Administrator role. Service principals are exempt from MFA but must use
certificate-based credentials with a maximum lifetime of 90 days.

## Encryption

Data is encrypted at rest with AES-256 using platform-managed keys by default.
Customer-managed keys are supported on Premium tier and require a key vault in the
same region as the protected resource. Key rotation is customer-initiated; Contoso
does not rotate customer-managed keys automatically.

Data in transit is protected with TLS 1.2 or higher. TLS 1.0 and 1.1 were retired in
all regions and connections using them are rejected.

## Certifications

Contoso maintains SOC 2 Type II, ISO 27001, and ISO 27018 certifications. Audit
reports are available to customers under NDA through the trust portal. HIPAA business
associate agreements are available on Enterprise plans only.

## Data residency

Customers may pin data residency to a geography at subscription creation. Residency
cannot be changed after creation; moving data between geographies requires creating a
new subscription and migrating workloads. Metadata such as resource names and tags
may be replicated globally for service operation and is not covered by residency
guarantees.

## Incident notification

Confirmed security incidents affecting customer data are notified to the account's
security contact within 72 hours of confirmation. Notification is sent by email and
posted to the service health dashboard.
