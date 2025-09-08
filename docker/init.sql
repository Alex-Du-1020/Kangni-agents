-- PostgreSQL 初始化脚本
-- 为 Kangni Agents 系统创建必要的数据库结构

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 创建用户表（如果需要）
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建查询历史表
CREATE TABLE IF NOT EXISTS query_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_email VARCHAR(255) NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    session_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建反馈表
CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_id UUID REFERENCES query_history(id) ON DELETE CASCADE,
    user_email VARCHAR(255) NOT NULL,
    feedback_type VARCHAR(20) CHECK (feedback_type IN ('like', 'dislike')),
    comment TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_query_history_user_email ON query_history(user_email);
CREATE INDEX IF NOT EXISTS idx_query_history_session_id ON query_history(session_id);
CREATE INDEX IF NOT EXISTS idx_query_history_created_at ON query_history(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_email ON feedback(user_email);

-- 插入示例数据（可选）
INSERT INTO users (email) VALUES ('admin@kangni.com') ON CONFLICT (email) DO NOTHING;

-- 设置时区
SET timezone = 'Asia/Shanghai';
