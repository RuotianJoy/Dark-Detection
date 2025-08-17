#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的LaTeX论文编译脚本
专门用于编译 interference_fringe_analysis_paper.tex
"""

import os
import subprocess
import sys
from pathlib import Path

def compile_paper():
    """
    编译论文LaTeX文件
    """
    tex_file = "interference_fringe_analysis_paper.tex"
    
    # 检查文件是否存在
    if not Path(tex_file).exists():
        print(f"❌ 错误: 找不到文件 {tex_file}")
        return False
    
    print(f"🔄 开始编译论文: {tex_file}")
    
    # 尝试不同的编译器
    compilers = ['xelatex', 'pdflatex', 'lualatex']
    
    for compiler in compilers:
        try:
            # 检查编译器是否可用
            result = subprocess.run([compiler, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✓ 使用编译器: {compiler}")
                
                # 编译命令
                cmd = [
                    compiler,
                    '-interaction=nonstopmode',
                    '-file-line-error',
                    tex_file
                ]
                
                print("📄 第一次编译...")
                result1 = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                print("📄 第二次编译（处理交叉引用）...")
                result2 = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                # 检查是否生成了PDF
                pdf_file = tex_file.replace('.tex', '.pdf')
                if Path(pdf_file).exists():
                    print(f"✅ 编译成功! 生成文件: {pdf_file}")
                    
                    # 清理临时文件
                    cleanup_temp_files()
                    return True
                else:
                    print(f"⚠️ {compiler} 编译未成功生成PDF")
                    if result2.returncode != 0:
                        print("错误信息:")
                        print(result2.stdout)
                        print(result2.stderr)
                    
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"❌ {compiler} 不可用")
            continue
    
    print("❌ 所有编译器都无法成功编译")
    print("\n💡 解决方案:")
    print("1. 安装LaTeX发行版 (TeX Live 或 MiKTeX)")
    print("2. 确保LaTeX编译器在系统PATH中")
    print("3. 检查LaTeX文件语法是否正确")
    return False

def cleanup_temp_files():
    """
    清理编译产生的临时文件
    """
    temp_extensions = ['.aux', '.log', '.out', '.toc', '.synctex.gz', '.fls', '.fdb_latexmk']
    base_name = "interference_fringe_analysis_paper"
    
    cleaned = []
    for ext in temp_extensions:
        temp_file = Path(base_name + ext)
        if temp_file.exists():
            try:
                temp_file.unlink()
                cleaned.append(temp_file.name)
            except Exception as e:
                print(f"⚠️ 无法删除 {temp_file}: {e}")
    
    if cleaned:
        print(f"🧹 已清理临时文件: {', '.join(cleaned)}")

def check_latex_installation():
    """
    检查LaTeX安装状态
    """
    print("🔍 检查LaTeX安装状态...")
    
    compilers = ['pdflatex', 'xelatex', 'lualatex']
    available = []
    
    for compiler in compilers:
        try:
            result = subprocess.run([compiler, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                available.append(compiler)
                print(f"✓ {compiler} 已安装")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"❌ {compiler} 未安装")
    
    if available:
        print(f"\n✅ 可用的LaTeX编译器: {', '.join(available)}")
        return True
    else:
        print("\n❌ 未找到任何LaTeX编译器")
        print("\n📚 安装指南:")
        print("Windows: 下载安装 MiKTeX (https://miktex.org/) 或 TeX Live")
        print("Linux: sudo apt-get install texlive-full")
        print("macOS: 下载安装 MacTeX (https://tug.org/mactex/)")
        return False

def main():
    print("📝 LaTeX论文编译工具")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        check_latex_installation()
        return
    
    # 检查LaTeX是否安装
    if not check_latex_installation():
        return
    
    print("\n" + "=" * 50)
    
    # 编译论文
    success = compile_paper()
    
    if success:
        print("\n🎉 论文编译完成!")
        print("📄 PDF文件已生成，可以查看编译结果")
    else:
        print("\n💥 编译失败，请检查LaTeX文件或安装")

if __name__ == '__main__':
    main()