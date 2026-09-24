# Security Statement

## Running User Suggestion

From a security perspective, do not use administrator accounts such as root to execute any commands. Follow the principle of least privilege.

## File Permission Control

- Set the umask value of the running system to 0027 or higher on the host (including the host machine) and in containers. This ensures that the default maximum permission for newly created folders is 750 and for newly created files is 640.
- Implement permission control and other security measures for sensitive content such as personal privacy data, business assets, source files, and various files saved during operator development. For example, permission control for the installation directory of this project and for input public data files. For the recommended permission settings, refer to [A-Maximum Recommended Permission Control for Files (Folders) in Various Scenarios](#a-maximum-recommended-permission-control-for-files-folders-in-various-scenarios).
- Control permissions during user installation and usage. For the recommended file permission settings, refer to [A-Maximum Recommended Permission Control for Files (Folders) in Various Scenarios](#a-maximum-recommended-permission-control-for-files-folders-in-various-scenarios).

## Build Security Statement

When compiling and installing this project from source code, you need to compile it yourself. Some intermediate files are generated during the compilation process. After compilation, control the permissions of the intermediate files to ensure file security.

## Runtime Security Statement

- **Important**: If you plan to run PyPTO operators in a real NPU environment, ensure that the Ascend HDK version is 25.5.0 or higher. HDK environments below the supported version are not within the PyPTO verification and support scope. Running abnormal operators may cause abnormal NPU status, affecting subsequent operator execution and causing issues such as AIC timeouts. In severe cases, you may need to restart the device or host to recover.
- Write operator calling scripts based on the runtime environment resource conditions. If the operator calling script does not match the resource conditions (for example, the space used for generating input data or benchmark calculation results exceeds the memory capacity limit, or the saved data in the script exceeds the disk space), errors may occur and cause the process to exit unexpectedly.
- When an operator runs abnormally, the process exits and prints error information. Locate the specific error cause based on the error prompt, including setting the operator to synchronous execution, viewing log files, and so on.

## Public Network Address Statement
The public network addresses included in the source code of this project are declared as follows:

| Type | Open Source Code Address | File Name | Public IP Address/Public URL Address/Domain Name/Email Address/Compressed File Address | Purpose Description |
| :------------: |:------------------------------------------------------------------------------------------:|:----------------------------------------------------------| :---------------------------------------------------------- |:-----------------------------------------|
| Dependency | Not applicable | cmake/third_party/json.cmake | https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/json-3.11.3.tar.gz | Download JSON source code from GitCode as a build dependency |
| Dependency | Not applicable | cmake/third_party/gtest.cmake | https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz | Download Google Test source code from GitCode as a build dependency |
| Dependency | Not applicable | cmake/third_party/secure_c.cmake | https://gitcode.com/cann-src-third-party/libboundscheck/releases/download/v1.1.16/libboundscheck-v1.1.16.tar.gz | Download libboundscheck source code from GitCode as a build dependency |

---

## Vulnerability Mechanism Description
[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### A-Maximum Recommended Permission Control for Files (Folders) in Various Scenarios

| Type | Linux Maximum Recommended Permission |
| -------------- | ---------------  |
| User home directory | 750 (rwxr-x---) |
| Program files (including script files, library files, and so on) | 550 (r-xr-x---) |
| Program file directory | 550 (r-xr-x---) |
| Configuration file | 640 (rw-r-----) |
| Configuration file directory | 750 (rwxr-x---) |
| Log file (completed or archived) | 440 (r--r-----) |
| Log file (being written) | 640 (rw-r-----) |
| Log file directory | 750 (rwxr-x---) |
| Debug file | 640 (rw-r-----) |
| Debug file directory | 750 (rwxr-x---) |
| Temporary file directory | 750 (rwxr-x---) |
| Maintenance upgrade file directory | 770 (rwxrwx---) |
| Business data file | 640 (rw-r-----) |
| Business data file directory | 750 (rwxr-x---) |
| Key component, private key, certificate, ciphertext file directory | 700 (rwx------) |
| Key component, private key, certificate, encrypted ciphertext | 600 (rw-------) |
| Encryption/decryption interface, encryption/decryption script | 500 (r-x------) |
