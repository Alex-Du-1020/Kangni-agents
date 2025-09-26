#!/usr/bin/env python3
"""
MinIO文件下载脚本
从MinIO服务器下载所有文件
"""
import os
import sys
from minio import Minio
from minio.error import S3Error
from urllib.parse import urlparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinIODownloader:
    def __init__(self, endpoint, access_key, secret_key, secure=False):
        """
        初始化MinIO客户端
        
        Args:
            endpoint: MinIO服务器地址 (例如: 158.58.50.45:9000)
            access_key: 访问密钥
            secret_key: 秘密密钥
            secure: 是否使用HTTPS
        """
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure
        )
        self.endpoint = endpoint
        logger.info(f"MinIO客户端初始化完成，服务器: {endpoint}")
    
    def list_buckets(self):
        """获取所有存储桶列表"""
        try:
            buckets = self.client.list_buckets()
            bucket_names = [bucket.name for bucket in buckets]
            logger.info(f"找到 {len(bucket_names)} 个存储桶: {bucket_names}")
            return bucket_names
        except S3Error as e:
            logger.error(f"获取存储桶列表失败: {e}")
            return []
    
    def list_objects(self, bucket_name, prefix=""):
        """
        获取指定存储桶中的所有对象列表
        
        Args:
            bucket_name: 存储桶名称
            prefix: 对象前缀过滤
        
        Returns:
            list: 对象信息列表
        """
        try:
            objects = []
            for obj in self.client.list_objects(bucket_name, prefix=prefix, recursive=True):
                objects.append({
                    'name': obj.object_name,
                    'size': obj.size,
                    'last_modified': obj.last_modified,
                    'etag': obj.etag
                })
            logger.info(f"存储桶 '{bucket_name}' 中找到 {len(objects)} 个对象")
            return objects
        except S3Error as e:
            logger.error(f"获取对象列表失败: {e}")
            return []
    
    def download_file(self, bucket_name, object_name, file_path):
        """
        下载单个文件
        
        Args:
            bucket_name: 存储桶名称
            object_name: 对象名称
            file_path: 本地保存路径
        
        Returns:
            bool: 下载是否成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 下载文件
            self.client.fget_object(bucket_name, object_name, file_path)
            logger.info(f"成功下载: {object_name} -> {file_path}")
            return True
        except S3Error as e:
            logger.error(f"下载文件失败 {object_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"下载文件异常 {object_name}: {e}")
            return False
    
    def download_all_files(self, download_dir="downloads", suffix_filter=None):
        """
        下载所有文件
        
        Args:
            download_dir: 下载目录
            suffix_filter: 文件后缀过滤 (例如: ['.pdf', '.docx', '.txt'])
        
        Returns:
            dict: 下载结果统计
        """
        results = {
            'total_buckets': 0,
            'total_objects': 0,
            'downloaded': 0,
            'failed': 0,
            'skipped': 0
        }
        
        # 获取所有存储桶
        buckets = self.list_buckets()
        results['total_buckets'] = len(buckets)
        
        if not buckets:
            logger.warning("没有找到任何存储桶")
            return results
        
        # 遍历每个存储桶
        for bucket_name in buckets:
            logger.info(f"处理存储桶: {bucket_name}")
            
            # 获取存储桶中的所有对象
            objects = self.list_objects(bucket_name)
            results['total_objects'] += len(objects)
            
            # 为每个存储桶创建子目录
            bucket_dir = os.path.join(download_dir, bucket_name)
            
            for obj in objects:
                object_name = obj['name']
                
                # 检查文件后缀过滤
                if suffix_filter:
                    file_ext = os.path.splitext(object_name)[1].lower()
                    if file_ext not in suffix_filter:
                        logger.debug(f"跳过文件 (后缀不匹配): {object_name}")
                        results['skipped'] += 1
                        continue
                
                # 构建本地文件路径
                local_path = os.path.join(bucket_dir, object_name)
                
                # 下载文件
                if self.download_file(bucket_name, object_name, local_path):
                    results['downloaded'] += 1
                else:
                    results['failed'] += 1
        
        return results
    
    def download_files_by_suffix(self, suffix_list, download_dir="downloads"):
        """
        根据后缀名下载文件
        
        Args:
            suffix_list: 后缀名列表 (例如: ['.pdf', '.docx', '.xlsx'])
            download_dir: 下载目录
        
        Returns:
            dict: 下载结果统计
        """
        logger.info(f"开始下载指定后缀的文件: {suffix_list}")
        return self.download_all_files(download_dir, suffix_list)

def main():
    """主函数"""
    # MinIO服务器配置
    ENDPOINT = "158.58.50.45:9000"
    ACCESS_KEY = "rag_flow"
    SECRET_KEY = "infini_rag_flow"
    SECURE = False  # 使用HTTP，不是HTTPS
    
    # 下载配置
    DOWNLOAD_DIR = "./minio_downloads"
    SUFFIX_FILTER = ['.pdf', '.docx', '.xlsx', '.txt', '.xls', '.doc']  # 可以根据需要修改
    
    try:
        # 初始化下载器
        downloader = MinIODownloader(
            endpoint=ENDPOINT,
            access_key=ACCESS_KEY,
            secret_key=SECRET_KEY,
            secure=SECURE
        )
        
        # 测试连接
        logger.info("测试MinIO连接...")
        buckets = downloader.list_buckets()
        if not buckets:
            logger.error("无法连接到MinIO服务器或没有存储桶")
            return
        
        logger.info("连接成功！")
        
        # # 方法1: 下载所有文件
        # logger.info("=" * 50)
        # logger.info("开始下载所有文件...")
        # results = downloader.download_all_files(DOWNLOAD_DIR)
        
        # # 打印结果
        # logger.info("=" * 50)
        # logger.info("下载完成！")
        # logger.info(f"存储桶数量: {results['total_buckets']}")
        # logger.info(f"对象总数: {results['total_objects']}")
        # logger.info(f"成功下载: {results['downloaded']}")
        # logger.info(f"下载失败: {results['failed']}")
        # logger.info(f"跳过文件: {results['skipped']}")
        
        # 方法2: 根据后缀下载特定文件
        logger.info("=" * 50)
        logger.info("开始下载指定后缀的文件...")
        suffix_results = downloader.download_files_by_suffix(SUFFIX_FILTER, "filtered_downloads")
        
        logger.info("=" * 50)
        logger.info("后缀过滤下载完成！")
        logger.info(f"存储桶数量: {suffix_results['total_buckets']}")
        logger.info(f"对象总数: {suffix_results['total_objects']}")
        logger.info(f"成功下载: {suffix_results['downloaded']}")
        logger.info(f"下载失败: {suffix_results['failed']}")
        logger.info(f"跳过文件: {suffix_results['skipped']}")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
