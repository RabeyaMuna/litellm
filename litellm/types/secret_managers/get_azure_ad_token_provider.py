from enum import Enum


class AzureCredentialType(str, Enum):
    ClientSecretCredential: str = "ClientSecretCredential"
    ManagedIdentityCredential: str = "ManagedIdentityCredential"
    CertificateCredential: str = "CertificateCredential"
