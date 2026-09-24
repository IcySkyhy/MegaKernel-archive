# Git 认证配置参考

> 由 SKILL.md 阶段 2 检测到认证缺失时加载。

## 认证方式对比

| 方式 | 安全性 | 持久性 | 配置难度 | 推荐场景 |
|------|--------|--------|----------|----------|
| **cache**（推荐） | ⭐⭐⭐⭐⭐ 内存 | 临时（可设超时） | 简单 | 容器/临时环境 |
| **store** | ⭐ 明文 | 持久 | 简单 | 个人开发机 |
| **SSH Key** | ⭐⭐⭐⭐⭐ 加密 | 持久 | 中等 | 长期开发 |
| **GITCODE_TOKEN + URL** | ⭐⭐ 环境变量 | 临时 | 简单 | CI/CD |
| **libsecret** | ⭐⭐⭐⭐⭐ 系统加密 | 持久 | 中等 | Linux 桌面 |

## 配置方法

### 方式 1: credential.helper cache（推荐容器环境）

```bash
git config --global credential.helper 'cache --timeout=604800'
# 首次 push 输入用户名和 Token，后续自动使用缓存（7 天有效）
```

### 方式 2: credential.helper store

```bash
git config --global credential.helper store
# 首次 push 输入用户名和 Token，保存到 ~/.git-credentials（明文）
```

### 方式 3: SSH Key

```bash
ssh-keygen -t ed25519 -C "your@email.com"
cat ~/.ssh/id_ed25519.pub  # 添加到 GitCode → Settings → SSH Keys
git remote set-url origin git@gitcode.com:<username>/pypto.git
```

### 方式 4: GITCODE_TOKEN + URL 内嵌

```bash
export GITCODE_TOKEN="your-token"
git remote set-url origin https://oauth2:${GITCODE_TOKEN}@gitcode.com/<username>/pypto.git
```

### 方式 5: libsecret（Linux 桌面）

```bash
sudo apt install libsecret-1-0 libsecret-1-dev libglib2.0-dev
cd /usr/share/doc/git/contrib/credential/libsecret && sudo make
git config --global credential.helper /usr/share/doc/git/contrib/credential/libsecret/git-credential-libsecret
```

## 验证认证

```bash
git push --dry-run origin <branch>
```

认证成功后才可继续后续阶段。
