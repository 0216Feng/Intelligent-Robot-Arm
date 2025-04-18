#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filepath: /D:/study/rdk_docker/data/control.py

"""
Ubuntu串口通信工具
用于查找可用串口设备并发送数据

使用方法:
    查看帮助: python3 control.py -h
    列出所有串口: python3 control.py --list
    发送数据: python3 control.py --port /dev/ttyUSB0 --data "你的命令"
"""

import serial
import serial.tools.list_ports
import argparse
import time
import logging
import os
import sys
import atexit

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("串口通信工具")

def list_serial_ports():
    """
    列出系统上所有可用的串口设备
    
    返回:
        list: 串口设备名称列表
    """
    ports = serial.tools.list_ports.comports()
    
    if not ports:
        logger.info("未检测到串口设备")
        return []
    
    logger.info("可用串口列表:")
    for i, port in enumerate(ports):
        # 显示详细信息
        device = port.device
        desc = port.description if port.description else "无描述"
        hwid = port.hwid if port.hwid else "无硬件ID"
        manufacturer = port.manufacturer if port.manufacturer else "未知厂商"
        
        logger.info(f"[{i+1}] {device}")
        logger.info(f"    描述: {desc}")
        logger.info(f"    硬件ID: {hwid}")
        logger.info(f"    厂商: {manufacturer}")
    
    return [port.device for port in ports]

def check_active_serial_ports():
    """
    检查当前正在使用的串口设备
    使用系统命令lsof查询
    """
    logger.info("检查正在使用的串口设备...")
    
    try:
        # 使用lsof命令查找正在使用的串口设备
        # 需要sudo权限才能查看所有进程
        use_sudo = os.geteuid() != 0
        sudo_prefix = "sudo " if use_sudo else ""
        
        cmd = f"{sudo_prefix}lsof | grep -E '/dev/tty|/dev/serial'"
        if use_sudo:
            logger.info("需要sudo权限查看所有进程使用的串口设备")
            logger.info(f"执行: {cmd}")
            logger.info("请输入您的密码（如果需要）:")
        
        # 执行命令
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0 and result.returncode != 1:  # grep返回1表示没有匹配
            logger.warning(f"命令执行失败: {result.stderr}")
            logger.info("您可以手动执行以下命令查看正在使用的串口:")
            logger.info("  sudo lsof | grep -E '/dev/tty|/dev/serial'")
            return
        
        output = result.stdout.strip()
        if not output:
            logger.info("未发现正在使用的串口设备")
            return
        
        logger.info("正在使用的串口设备:")
        print(output)
        
    except Exception as e:
        logger.error(f"检查正在使用的串口设备时出错: {e}")
        logger.info("您可以手动执行以下命令查看正在使用的串口:")
        logger.info("  sudo lsof | grep -E '/dev/tty|/dev/serial'")

# 添加全局变量以保存串口连接
_serial_connection = None
_last_used_port = None
_last_used_baud = None

def send_data_to_serial(port, data, baud_rate=115200, timeout=1, hex_mode=False, keep_open=True):
    """
    向指定串口发送数据
    
    参数:
        port (str): 串口设备名，如 '/dev/ttyUSB0'
        data (str): 要发送的数据
        baud_rate (int): 波特率
        timeout (int): 超时时间(秒)
        hex_mode (bool): 是否以十六进制模式发送
        keep_open (bool): 发送后是否保持串口开启
    
    返回:
        bool: 操作是否成功
    """
    global _serial_connection, _last_used_port, _last_used_baud
    
    try:
        # 检查是否可以复用现有连接
        if _serial_connection and _serial_connection.is_open and _last_used_port == port and _last_used_baud == baud_rate:
            ser = _serial_connection
            logger.info(f"复用已打开的串口 {port}")
        else:
            # 如果之前有连接，先关闭
            if _serial_connection and _serial_connection.is_open:
                try:
                    _serial_connection.close()
                    logger.info(f"关闭之前的串口 {_last_used_port}")
                except:
                    pass
            
            # 打开新的串口连接
            ser = serial.Serial(
                port=port,
                baudrate=baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout
            )
            
            if not ser.is_open:
                ser.open()
                
            logger.info(f"成功打开串口 {port}, 波特率: {baud_rate}")
            
            # 更新全局变量
            _serial_connection = ser
            _last_used_port = port
            _last_used_baud = baud_rate
            
            # 等待2秒，让设备准备好
            logger.info("等待2秒，确保设备准备就绪...")
            time.sleep(2)
        
        # 准备发送的数据
        if hex_mode:
            # 处理十六进制输入
            try:
                # 移除空格和0x前缀
                clean_data = data.replace(" ", "").replace("0x", "").replace(",", "")
                # 将十六进制字符串转换为字节
                send_bytes = bytes.fromhex(clean_data)
            except ValueError as e:
                logger.error(f"十六进制数据格式错误: {e}")
                if not keep_open:
                    ser.close()
                    _serial_connection = None
                return False
        else:
            # 普通字符串，编码为bytes
            send_bytes = data.encode('utf-8')
        
        # 显示要发送的数据
        logger.info(f"发送数据: '{data}'")
        logger.info(f"十六进制形式: {' '.join([f'0x{b:02X}' for b in send_bytes])} ({len(send_bytes)} 字节)")
        
        # 发送数据
        bytes_sent = ser.write(send_bytes)
        logger.info(f"已发送 {bytes_sent} 字节")
        
        # 等待数据发送完成
        ser.flush()
        
        # 尝试读取回复(如果有)
        time.sleep(0.5)  # 给设备一点响应时间
        response_bytes = bytearray()
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if ser.in_waiting:
                response_bytes.extend(ser.read(ser.in_waiting))
                time.sleep(0.1)  # 短暂等待更多数据
            else:
                break
        
        if response_bytes:
            logger.info(f"收到回复: {' '.join([f'0x{b:02X}' for b in response_bytes])} ({len(response_bytes)} 字节)")
            
            # 尝试以不同编码解码响应
            try:
                # 尝试UTF-8解码
                utf8_text = response_bytes.decode('utf-8', errors='replace')
                logger.info(f"UTF-8解码: {utf8_text}")
            except UnicodeDecodeError:
                pass
                
            try:
                # 尝试ASCII解码(仅适用于可打印字符)
                if all(32 <= b <= 126 for b in response_bytes):
                    ascii_text = response_bytes.decode('ascii', errors='replace')
                    logger.info(f"ASCII解码: {ascii_text}")
            except:
                pass
        
        # 根据参数决定是否关闭串口
        if not keep_open:
            ser.close()
            _serial_connection = None
            logger.info("串口已关闭")
        else:
            logger.info("保持串口开启状态")
            
        return True
        
    except serial.SerialException as e:
        logger.error(f"串口通信错误: {e}")
        _serial_connection = None
        return False
    except Exception as e:
        logger.error(f"发生错误: {e}")
        _serial_connection = None
        return False

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='Ubuntu串口通信工具 - 向串口设备发送数据')
    parser.add_argument('--port', '-p', default='/dev/ttyUSB0', help='串口设备名，如 /dev/ttyUSB0')
    parser.add_argument('--baud', '-b', type=int, default=115200, help='波特率，默认: 115200')
    parser.add_argument('--data', '-d', help='要发送的数据字符串')
    parser.add_argument('--hex', '-x', action='store_true', help='以十六进制模式发送数据')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有可用串口')
    parser.add_argument('--check', '-c', action='store_true', help='检查正在使用的串口设备')
    parser.add_argument('--file', '-f', help='从文件读取要发送的数据')
    parser.add_argument('--timeout', '-t', type=float, default=1.0, help='超时时间(秒)，默认: 1.0')
    parser.add_argument('--keep-open', '-k', action='store_true', help='发送后保持串口开启状态')
    
    args = parser.parse_args()
    
    # 列出所有串口
    if args.list:
        list_serial_ports()
        return
    
    # 检查正在使用的串口
    if args.check:
        check_active_serial_ports()
        return
    
    # 检查是否指定了串口
    if not args.port:
        available_ports = list_serial_ports()
        if not available_ports:
            logger.error("未找到可用串口，请检查设备连接")
            return
        
        if len(available_ports) == 1:
            args.port = available_ports[0]
            logger.info(f"自动选择唯一可用串口: {args.port}")
        else:
            try:
                port_idx = int(input(f"请选择串口 (1-{len(available_ports)}): ")) - 1
                if 0 <= port_idx < len(available_ports):
                    args.port = available_ports[port_idx]
                else:
                    logger.error("无效的选择")
                    return
            except ValueError:
                logger.error("无效的输入")
                return
    
    # 获取要发送的数据
    data_to_send = None
    
    if args.file:
        try:
            with open(args.file, 'r') as f:
                data_to_send = f.read()
            logger.info(f"从文件 {args.file} 读取数据")
        except Exception as e:
            logger.error(f"读取文件错误: {e}")
            return
    elif args.data:
        data_to_send = args.data
    else:
        logger.info("未提供数据，请输入要发送的数据 (按回车发送，Ctrl+D 结束输入):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        data_to_send = "\n".join(lines)
    
    if not data_to_send:
        logger.error("未提供要发送的数据")
        return
    
    # 发送数据
    send_data_to_serial(args.port, data_to_send, args.baud, args.timeout, 
                       hex_mode=args.hex, keep_open=args.keep_open)

# 定义退出时的清理函数
def cleanup():
    global _serial_connection
    if _serial_connection and _serial_connection.is_open:
        try:
            _serial_connection.close()
            logger.info("程序退出，关闭串口")
        except:
            pass

# 注册退出函数
atexit.register(cleanup)

if __name__ == "__main__":
    main()