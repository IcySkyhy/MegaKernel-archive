# 项目文档

## 简介

此目录提供[PyPTO文档中心](https://pypto.gitcode.com)的源文件信息，包括环境部署、算子开发、API参考等。

## 贡献

欢迎您参与文档贡献！详细请参考[文档贡献指南](./CONTRIBUTION_DOC.md)，请您务必遵守文档写作规范，并按照流程规则提交。审核通过后，将会在本项目docs目录和文档中心网页中呈现。如果您对文档有任何意见或建议，请在Issues中提交。

## 目录说明

关键目录结构如下：

```txt
docs/
├── zh/                         # 中文文档
│   ├── install                 # 环境部署
│   ├── invocation              # 样例运行
│   ├── tutorials               # PyPTO算子开发
│   └── api                     # PyPTO API
├── CONTRIBUTION_DOC.md         # 文档贡献指南
└── README.md                   # 本文件
```

## 文档构建

 PyPTO文档可由Sphinx工具生成，当文档PR合入后，将自动触发文档构建。同时也支持本地构建，本地构建文档前需要安装必要模块，以下是具体步骤。

1. 下载PyPTO仓代码。

   ```bash
   git clone https://gitcode.com/cann/pypto.git
   ```

2. 进入docs目录并安装该目录下`requirements.txt`所需依赖。

   ```bash
   cd docs
   pip install -r requirements.txt
   ```

3. 在docs目录下执行如下命令进行文档构建。

   ```bash
   make html
   ```

4. 构建完成后会新建_build/html目录，执行如下命令启动HTTP服务器以提供文档服务。

   ```bash
   cd _build/html
   python3 -m http.server 8000
   ```

   默认端口8000，也可自行指定端口。

5. 在浏览器中访问`http://localhost:8000`查看文档。
