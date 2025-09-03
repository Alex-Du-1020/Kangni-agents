import asyncio
import logging
import os
from mysql.connector import connect, Error
from mcp.server.lowlevel import Server
from mcp.types import Resource, Tool, TextContent
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl
from dotenv import load_dotenv

# 日志相关配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mysql_mcp_server")


# 获取数据库配置
def get_db_config():
    # 加载 .env 文件中的环境变量到系统环境变量中
    load_dotenv()
    # 从环境变量中获取数据库配置
    config = {
        "host": os.getenv("MYSQL_HOST"),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE"),
        "port": os.getenv("MYSQL_PORT"),
        # 添加认证插件配置，解决 caching_sha2_password 不支持的问题
        "auth_plugin": "mysql_native_password",
        # 添加其他连接参数以提高兼容性
        "charset": "utf8mb4",
        "use_unicode": True,
        "autocommit": True
    }

    # 检查是否存在配置中的关键字段
    # 记录错误信息，提示用户检查环境变量
    if not all([config["user"], config["password"], config["database"]]):
        logger.error("Missing required database configuration. Please check environment variables:")
        logger.error("MYSQL_USER, MYSQL_PASSWORD, and MYSQL_DATABASE are required")
        # 抛出一个 ValueError 异常，终止函数的执行
        raise ValueError("Missing required database configuration")

    # 配置完整，则返回包含数据库配置的字典config
    return config


# 实例化Server
mcp = Server("mysql_mcp_server")


# 声明 list_resources 函数为一个资源列表接口
# 列出 MySQL 数据库中的表并将其作为资源返回
@mcp.list_resources()
async def list_resources() -> list[Resource]:
    # 获取数据库配置
    config = get_db_config()
    try:
        # 连接数据库
        # with 语句用于确保连接在使用完成后自动关闭（即使发生异常）
        with connect(**config) as conn:
            # 创建一个数据库游标 cursor，用于执行 SQL 查询
            # with 确保游标在操作完成后自动关闭
            with conn.cursor() as cursor:
                # 执行 SQL 查询 SHOW TABLES，它列出当前数据库中的所有表
                cursor.execute("SHOW TABLES")
                # 使用 fetchall 方法获取查询结果，返回一个包含所有表名的列表
                tables = cursor.fetchall()
                logger.info(f"Found tables: {tables}")

                # 初始化一个空列表 resources，用于存储 Resource 对象
                resources = []
                # 遍历 tables 列表，每次迭代处理一个表名
                for table in tables:
                    # 创建一个 Resource 对象
                    # 填充其属性：
                    # uri: 表的唯一资源标识符
                    # name: 资源的名称
                    # mimeType: MIME 类型，表示资源的数据类型
                    # description: 描述信息
                    cursor.execute(f"SHOW COLUMNS FROM {table}")
                    columns = [row[0] for row in cursor.fetchall()]

                    resources.append(
                        Resource(
                            uri=f"mysql://{table[0]}/data",
                            name=f"Table: {table[0]}",
                            mimeType="text/plain",
                            description=f"Data in table: table {table[0]} columns: {', '.join(columns)}"
                        )
                    )
                # 在成功获取表信息并构造 Resource 对象列表后，返回 resources
                return resources
    # 发生异常，返回一个空列表，表明未能成功获取任何资源
    except Error as e:
        logger.error(f"Failed to list resources: {str(e)}")
        return []
        
def get_table_schema():
    config = get_db_config()
    schema_info = []
    with connect(**config) as conn:
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES")
            tables = [row[0] for row in cur.fetchall()]
            for table in tables:
                cur.execute(f"SHOW COLUMNS FROM {table}")
                columns = [row[0] for row in cur.fetchall()]
                schema_info.append(f"表 {table} 字段: {', '.join(columns)}")
            return '\n'.join(schema_info)


# 声明 read_resource 函数为一个读取资源的接口
# 根据传入的 URI 读取表的内容
# uri: 表示资源的 URI，类型为 AnyUrl，确保输入是一个合法的 URL
@mcp.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    # 获取数据库配置
    config = get_db_config()
    # 将 uri 对象转换为字符串形式，存储在 uri_str 变量中
    uri_str = str(uri)
    logger.info(f"Reading resource: {uri_str}")

    # 检查 uri_str 是否以 "mysql://" 开头，确保 URI 符合预期的 MySQL 资源格式
    if not uri_str.startswith("mysql://"):
        # 如果不符合，则抛出 ValueError 异常，提示 URI 格式无效
        raise ValueError(f"Invalid URI scheme: {uri_str}")

    # 将 URI 中 "mysql://" 部分去掉，并按照 '/' 分割为多个部分
    parts = uri_str[8:].split('/')
    # parts[0] 是表名，存储到变量 table 中
    table = parts[0]

    try:
        # 使用数据库连接库的 connect 方法，使用 config 中的配置参数连接到数据库
        # with 确保在连接关闭时资源被正确释放
        with connect(**config) as conn:
            # 创建一个数据库游标 cursor，用于执行 SQL 查询
            # with 确保游标在操作完成后自动关闭
            with conn.cursor() as cursor:
                # 执行 SQL 查询，读取表中的前 100 条记录
                # 注意：此处直接使用 table 变量拼接查询字符串，有潜在的 SQL 注入风险
                cursor.execute(f"SELECT * FROM {table} LIMIT 100")
                # 获取表的列名：
                # cursor.description 返回查询结果的列描述信息
                # 列表推导式提取每列的名称（desc[0]）
                columns = [desc[0] for desc in cursor.description]
                # 使用 fetchall 方法获取查询结果的所有行
                # 返回值 rows 是一个包含多行数据的列表，每行是一个元组
                rows = cursor.fetchall()
                # 使用列表推导式将每行数据转换为逗号分隔的字符串
                # map(str, row) 将行中的每个元素转换为字符串
                # ",".join(...) 将字符串连接成一行数据
                # 结果存储在 result 列表中
                result = [",".join(map(str, row)) for row in rows]
                # 将列名（作为第一行）和数据行组合成最终字符串返回
                return "\n".join([",".join(columns)] + result)
    # 捕获数据库连接或查询期间发生的任何异常
    except Error as e:
        logger.error(f"Database error reading resource {uri}: {str(e)}")
        raise RuntimeError(f"Database error: {str(e)}")


# 声明 list_tools 函数为一个列出工具的接口
# 列出可用的 MySQL 工具
@mcp.list_tools()
async def list_tools() -> list[Tool]:
    logger.info("Listing tools...")
    # 函数返回一个列表，其中包含多个 Tool 对象
    # 每个 Tool 对象代表一个工具，其属性定义了工具的功能和输入要求
    return [
        Tool(
            # 工具的名称
            name="execute_sql",
            # 工具的描述
            description="Execute an SQL query on the MySQL server",
            # 定义了工具的输入模式（Schema），用于描述输入数据的格式和要求
            inputSchema={
                # 定义输入为一个 JSON 对象
                "type": "object",
                # 定义输入对象的属性
                "properties": {
                    # 指明此属性存储要执行的 SQL 查询
                    "query": {
                        "type": "string",
                        "description": "The SQL query to execute"
                    }
                },
                # 列出输入对象的必需属性
                "required": ["query"]
            }
        ),
        Tool(
            name="get_schema_info",
            description="Get database schema information including all tables and their columns",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
    ]


# 声明 call_tool 函数为一个工具调用的接口
# 根据传入的工具名称和参数执行相应的 SQL 命令
# name: 工具的名称（字符串），指定要调用的工具
# arguments: 一个字典，包含工具所需的参数
@mcp.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # 获取数据库配置
    config = get_db_config()
    logger.info(f"Calling tool: {name} with arguments: {arguments}")

    # 根据工具名称执行相应的操作
    if name == "execute_sql":
        return await _execute_sql(config, arguments)
    elif name == "get_schema_info":
        return await _get_schema_info(config)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _execute_sql(config: dict, arguments: dict) -> list[TextContent]:
    """执行SQL查询的工具函数"""
    query = arguments.get("query")
    if not query:
        raise ValueError("Query is required")

    try:
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)

                if query.strip().upper().startswith("SHOW TABLES"):
                    tables = cursor.fetchall()
                    result = ["Tables_in_" + config["database"]]
                    result.extend([table[0] for table in tables])
                    return [TextContent(type="text", text="\n".join(result))]

                elif query.strip().upper().startswith("SELECT"):
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    result = [",".join(map(str, row)) for row in rows]
                    return [TextContent(type="text", text="\n".join([",".join(columns)] + result))]

                else:
                    conn.commit()
                    return [TextContent(type="text", text=f"Query executed successfully. Rows affected: {cursor.rowcount}")]
    except Error as e:
        logger.error(f"Error executing SQL '{query}': {e}")
        return [TextContent(type="text", text=f"Error executing query: {str(e)}")]


async def _get_schema_info(config: dict) -> list[TextContent]:
    """获取数据库schema信息的工具函数"""
    try:
        with connect(**config) as conn:
            with conn.cursor() as cursor:
                # 获取所有表名
                cursor.execute("SHOW TABLES")
                tables = [row[0] for row in cursor.fetchall()]
                
                schema_info = []
                for table in tables:
                    # 获取每个表的列信息
                    cursor.execute(f"SHOW COLUMNS FROM {table}")
                    columns = [row[0] for row in cursor.fetchall()]
                    schema_info.append(f"表 {table} 字段: {', '.join(columns)}")
                
                result = '\n'.join(schema_info)
                return [TextContent(type="text", text=result)]
                
    except Error as e:
        logger.error(f"Error getting schema info: {e}")
        return [TextContent(type="text", text=f"Error getting schema info: {str(e)}")]


# 启动 MCP服务器
async def main():
    logger.info("Starting MySQL MCP server...")
    # 获取数据库连接的配置信息
    config = get_db_config()
    logger.info(f"Database config: {config['host']}/{config['database']} as {config['user']}")

    # 启动 stdio_server，通过标准输入/输出（stdin/stdout）与客户端通信
    # async with 是异步上下文管理器，确保 stdio_server 资源在使用完成后自动清理
    # 返回的 (read_stream, write_stream) 是两个流对象
    # read_stream: 用于从客户端读取输入的流对象
    # write_stream: 用于向客户端发送输出的流对象
    async with stdio_server() as (read_stream, write_stream):
        try:
            # 异步运行 MCP 应用程序
            await mcp.run(
                read_stream,
                write_stream,
                # 用于初始化应用程序的选项，通常包含配置或上下文信息
                mcp.create_initialization_options()
            )
        # 捕获运行 mcp.run() 时发生的所有异常
        except Exception as e:
            logger.error(f"Server error: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    asyncio.run(main())