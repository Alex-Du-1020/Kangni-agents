#!/bin/bash

echo "构建Docker镜像..."
docker build -t kangni-agents .

echo "运行RAG测试..."
docker run --rm kangni-agents

echo "测试完成！"