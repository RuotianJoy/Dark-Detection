#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaTeX文件编译器
用于编译LaTeX文件并生成PDF输出
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_latex_installation():
    """
    检查LaTeX是否已安装
    """
    try:
        # 检查pdflatex
        result = subprocess.run(['pdflatex', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ pdflatex 已安装")
            return 'pdflatex'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    try:
        # 检查xelatex（支持中文更好）
        result = subprocess.run(['xelatex', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ xelatex 已安装")
            return 'xelatex'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    try:
        # 检查lualatex
        result = subprocess.run(['lualatex', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ lualatex 已安装")
            return 'lualatex'
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return None

def compile_latex(tex_file, compiler='auto', output_dir=None, clean=True):
    """
    编译LaTeX文件
    
    Args:
        tex_file (str): LaTeX文件路径
        compiler (str): 编译器选择 ('auto', 'pdflatex', 'xelatex', 'lualatex')
        output_dir (str): 输出目录
        clean (bool): 是否清理临时文件
    
    Returns:
        bool: 编译是否成功
    """
    tex_path = Path(tex_file)
    
    if not tex_path.exists():
        print(f"❌ 错误: 文件 {tex_file} 不存在")
        return False
    
    if not tex_path.suffix.lower() == '.tex':
        print(f"❌ 错误: {tex_file} 不是LaTeX文件")
        return False
    
    # 自动选择编译器
    if compiler == 'auto':
        compiler = check_latex_installation()
        if not compiler:
            print("❌ 错误: 未找到LaTeX编译器")
            print("请安装TeX Live或MiKTeX")
            return False
    
    # 设置输出目录
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_option = f'-output-directory={output_dir}'
    else:
        output_option = ''
    
    # 构建编译命令
    base_cmd = [
        compiler,
        '-interaction=nonstopmode',  # 非交互模式
        '-file-line-error',          # 显示文件行错误
        '-synctex=1',               # 生成同步文件
    ]
    
    if output_option:
        base_cmd.append(output_option)
    
    base_cmd.append(str(tex_path))
    
    print(f"🔄 开始编译: {tex_file}")
    print(f"📝 使用编译器: {compiler}")
    
    # 切换到tex文件所在目录
    original_cwd = os.getcwd()
    os.chdir(tex_path.parent)
    
    try:
        # 第一次编译
        print("📄 第一次编译...")
        result1 = subprocess.run(base_cmd, capture_output=True, text=True, timeout=120)
        
        if result1.returncode != 0:
            print("❌ 第一次编译失败:")
            print(result1.stdout)
            print(result1.stderr)
            return False
        
        # 第二次编译（处理交叉引用）
        print("📄 第二次编译（处理交叉引用）...")
        result2 = subprocess.run(base_cmd, capture_output=True, text=True, timeout=120)
        
        if result2.returncode != 0:
            print("⚠️ 第二次编译有警告，但可能已生成PDF")
            print(result2.stdout)
        
        # 检查PDF是否生成
        pdf_name = tex_path.stem + '.pdf'
        if output_dir:
            pdf_path = Path(output_dir) / pdf_name
        else:
            pdf_path = tex_path.parent / pdf_name
        
        if pdf_path.exists():
            print(f"✅ 编译成功! PDF已生成: {pdf_path}")
            
            # 清理临时文件
            if clean:
                cleanup_files(tex_path.parent, tex_path.stem)
            
            return True
        else:
            print("❌ 编译失败: 未生成PDF文件")
            return False
    
    except subprocess.TimeoutExpired:
        print("❌ 编译超时")
        return False
    except Exception as e:
        print(f"❌ 编译过程中出现错误: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def cleanup_files(directory, basename):
    """
    清理LaTeX编译产生的临时文件
    """
    temp_extensions = ['.aux', '.log', '.out', '.toc', '.lof', '.lot', 
                      '.fls', '.fdb_latexmk', '.synctex.gz', '.bbl', '.blg']
    
    directory = Path(directory)
    cleaned_files = []
    
    for ext in temp_extensions:
        temp_file = directory / (basename + ext)
        if temp_file.exists():
            try:
                temp_file.unlink()
                cleaned_files.append(temp_file.name)
            except Exception as e:
                print(f"⚠️ 无法删除临时文件 {temp_file}: {e}")
    
    if cleaned_files:
        print(f"🧹 已清理临时文件: {', '.join(cleaned_files)}")

def install_latex_guide():
    """
    显示LaTeX安装指南
    """
    print("\n📚 LaTeX安装指南:")
    print("\n🪟 Windows:")
    print("  1. 下载并安装 MiKTeX: https://miktex.org/download")
    print("  2. 或下载并安装 TeX Live: https://tug.org/texlive/")
    
    print("\n🐧 Linux (Ubuntu/Debian):")
    print("  sudo apt-get install texlive-full")
    
    print("\n🍎 macOS:")
    print("  1. 下载并安装 MacTeX: https://tug.org/mactex/")
    print("  2. 或使用 Homebrew: brew install --cask mactex")
    
    print("\n💡 推荐安装完整版本以获得所有宏包支持")

def main():
    parser = argparse.ArgumentParser(description='LaTeX文件编译器')
    parser.add_argument('tex_file', help='要编译的LaTeX文件路径')
    parser.add_argument('-c', '--compiler', 
                       choices=['auto', 'pdflatex', 'xelatex', 'lualatex'],
                       default='auto', help='选择编译器 (默认: auto)')
    parser.add_argument('-o', '--output', help='输出目录')
    parser.add_argument('--no-clean', action='store_true', 
                       help='不清理临时文件')
    parser.add_argument('--check', action='store_true', 
                       help='检查LaTeX安装')
    parser.add_argument('--install-guide', action='store_true', 
                       help='显示LaTeX安装指南')
    
    args = parser.parse_args()
    
    if args.install_guide:
        install_latex_guide()
        return
    
    if args.check:
        compiler = check_latex_installation()
        if compiler:
            print(f"✅ LaTeX已正确安装，可用编译器: {compiler}")
        else:
            print("❌ 未找到LaTeX编译器")
            install_latex_guide()
        return
    
    # 编译LaTeX文件
    success = compile_latex(
        tex_file=args.tex_file,
        compiler=args.compiler,
        output_dir=args.output,
        clean=not args.no_clean
    )
    
    if success:
        print("\n🎉 编译完成!")
        sys.exit(0)
    else:
        print("\n💥 编译失败!")
        sys.exit(1)

if __name__ == '__main__':
    # 如果直接运行，编译当前目录下的LaTeX文件
    if len(sys.argv) == 1:
        # 查找当前目录下的.tex文件
        tex_files = list(Path('.').glob('*.tex'))
        if tex_files:
            print(f"找到LaTeX文件: {[str(f) for f in tex_files]}")
            for tex_file in tex_files:
                print(f"\n编译文件: {tex_file}")
                compile_latex(str(tex_file))
        else:
            print("当前目录下没有找到.tex文件")
            print("\n使用方法:")
            print("  python latex_compiler.py <tex_file>")
            print("  python latex_compiler.py --help")
    else:
        main()