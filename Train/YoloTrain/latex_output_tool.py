#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaTeX文件输出工具
支持多种输出格式：HTML预览、纯文本、Markdown等
"""

import re
import os
import sys
from pathlib import Path
import argparse
import subprocess
import shutil

class LaTeXOutputTool:
    def __init__(self):
        self.supported_formats = ['html', 'txt', 'md', 'pdf']
    
    def check_latex_installation(self):
        """检查LaTeX编译器安装状态"""
        compilers = ['pdflatex', 'xelatex', 'lualatex']
        available = []
        
        for compiler in compilers:
            if shutil.which(compiler):
                available.append(compiler)
        
        return available
    
    def compile_to_pdf(self, tex_file, output_dir=None, compiler='pdflatex'):
        """编译LaTeX文件为PDF"""
        if not Path(tex_file).exists():
            print(f"❌ 错误: 文件 {tex_file} 不存在")
            return False
        
        available_compilers = self.check_latex_installation()
        if not available_compilers:
            print("❌ 未找到LaTeX编译器")
            return False
        
        if compiler not in available_compilers:
            compiler = available_compilers[0]
            print(f"⚠️  使用 {compiler} 编译器")
        
        try:
            # 设置输出目录
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                cmd = [compiler, f'-output-directory={output_dir}', tex_file]
            else:
                cmd = [compiler, tex_file]
            
            # 编译两次以处理交叉引用
            print(f"🔄 第一次编译...")
            result1 = subprocess.run(cmd, capture_output=True, text=True)
            
            print(f"🔄 第二次编译...")
            result2 = subprocess.run(cmd, capture_output=True, text=True)
            
            if result2.returncode == 0:
                pdf_file = Path(tex_file).stem + '.pdf'
                if output_dir:
                    pdf_file = os.path.join(output_dir, pdf_file)
                print(f"✅ PDF编译成功: {pdf_file}")
                return True
            else:
                print(f"❌ 编译失败: {result2.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 编译错误: {e}")
            return False
    
    def convert_to_html(self, tex_file, output_file=None):
        """转换为HTML格式"""
        from latex_to_html import latex_to_html
        return latex_to_html(tex_file, output_file)
    
    def convert_to_text(self, tex_file, output_file=None):
        """转换为纯文本格式"""
        if not Path(tex_file).exists():
            print(f"❌ 错误: 文件 {tex_file} 不存在")
            return False
        
        with open(tex_file, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        
        # 提取文档内容
        content_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
        if content_match:
            content = content_match.group(1)
        else:
            content = latex_content
        
        # 提取标题和作者
        title_match = re.search(r'\\title\{(.*?)\}', latex_content, re.DOTALL)
        author_match = re.search(r'\\author\{(.*?)\}', latex_content, re.DOTALL)
        
        text_content = ""
        if title_match:
            text_content += f"标题: {self.clean_latex_text(title_match.group(1))}\n\n"
        if author_match:
            text_content += f"作者: {self.clean_latex_text(author_match.group(1))}\n\n"
        
        # 转换内容
        text_content += self.latex_to_text(content)
        
        # 确定输出文件名
        if output_file is None:
            output_file = Path(tex_file).stem + '.txt'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"✅ 文本转换完成: {output_file}")
        return True
    
    def convert_to_markdown(self, tex_file, output_file=None):
        """转换为Markdown格式"""
        if not Path(tex_file).exists():
            print(f"❌ 错误: 文件 {tex_file} 不存在")
            return False
        
        with open(tex_file, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        
        # 提取文档内容
        content_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
        if content_match:
            content = content_match.group(1)
        else:
            content = latex_content
        
        # 提取标题和作者
        title_match = re.search(r'\\title\{(.*?)\}', latex_content, re.DOTALL)
        author_match = re.search(r'\\author\{(.*?)\}', latex_content, re.DOTALL)
        
        md_content = ""
        if title_match:
            md_content += f"# {self.clean_latex_text(title_match.group(1))}\n\n"
        if author_match:
            md_content += f"**作者:** {self.clean_latex_text(author_match.group(1))}\n\n"
        
        # 转换内容
        md_content += self.latex_to_markdown(content)
        
        # 确定输出文件名
        if output_file is None:
            output_file = Path(tex_file).stem + '.md'
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"✅ Markdown转换完成: {output_file}")
        return True
    
    def latex_to_text(self, content):
        """将LaTeX内容转换为纯文本"""
        # 处理摘要
        content = re.sub(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', 
                        r'摘要:\n\1\n', content, flags=re.DOTALL)
        
        # 处理章节标题
        content = re.sub(r'\\section\*?\{(.*?)\}', r'\n\1\n' + '='*50 + '\n', content)
        content = re.sub(r'\\subsection\*?\{(.*?)\}', r'\n\1\n' + '-'*30 + '\n', content)
        content = re.sub(r'\\subsubsection\*?\{(.*?)\}', r'\n\1\n', content)
        
        # 处理列表
        content = re.sub(r'\\begin\{itemize\}(.*?)\\end\{itemize\}', 
                        self.convert_itemize_text, content, flags=re.DOTALL)
        content = re.sub(r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}', 
                        self.convert_enumerate_text, content, flags=re.DOTALL)
        
        # 清理LaTeX命令
        content = self.clean_latex_text(content)
        
        return content
    
    def latex_to_markdown(self, content):
        """将LaTeX内容转换为Markdown"""
        # 处理摘要
        content = re.sub(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', 
                        r'## 摘要\n\n\1\n', content, flags=re.DOTALL)
        
        # 处理章节标题
        content = re.sub(r'\\section\*?\{(.*?)\}', r'\n## \1\n', content)
        content = re.sub(r'\\subsection\*?\{(.*?)\}', r'\n### \1\n', content)
        content = re.sub(r'\\subsubsection\*?\{(.*?)\}', r'\n#### \1\n', content)
        
        # 处理文本格式
        content = re.sub(r'\\textbf\{(.*?)\}', r'**\1**', content)
        content = re.sub(r'\\textit\{(.*?)\}', r'*\1*', content)
        content = re.sub(r'\\emph\{(.*?)\}', r'*\1*', content)
        
        # 处理列表
        content = re.sub(r'\\begin\{itemize\}(.*?)\\end\{itemize\}', 
                        self.convert_itemize_md, content, flags=re.DOTALL)
        content = re.sub(r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}', 
                        self.convert_enumerate_md, content, flags=re.DOTALL)
        
        # 处理数学公式
        content = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', 
                        r'\n$$\1$$\n', content, flags=re.DOTALL)
        
        # 清理LaTeX命令
        content = self.clean_latex_text(content)
        
        return content
    
    def convert_itemize_text(self, match):
        """转换无序列表为文本"""
        items = match.group(1)
        items = re.sub(r'\\item\s*', '• ', items)
        return f'\n{items}\n'
    
    def convert_enumerate_text(self, match):
        """转换有序列表为文本"""
        items = match.group(1)
        items = re.sub(r'\\item\s*', lambda m: f'{m.start()//10 + 1}. ', items)
        return f'\n{items}\n'
    
    def convert_itemize_md(self, match):
        """转换无序列表为Markdown"""
        items = match.group(1)
        items = re.sub(r'\\item\s*', '- ', items)
        return f'\n{items}\n'
    
    def convert_enumerate_md(self, match):
        """转换有序列表为Markdown"""
        items = match.group(1)
        counter = 1
        def replace_item(m):
            nonlocal counter
            result = f'{counter}. '
            counter += 1
            return result
        items = re.sub(r'\\item\s*', replace_item, items)
        return f'\n{items}\n'
    
    def clean_latex_text(self, text):
        """清理LaTeX文本中的特殊命令"""
        # 移除常见的LaTeX命令
        text = re.sub(r'\\[a-zA-Z]+\*?\{.*?\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
        text = re.sub(r'\{|\}', '', text)
        text = re.sub(r'\\\\', '\n', text)
        text = re.sub(r'\\&', '&', text)
        text = re.sub(r'\\%', '%', text)
        text = re.sub(r'\\\$', '$', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()
    
    def process_file(self, tex_file, output_format, output_file=None, **kwargs):
        """处理LaTeX文件"""
        if output_format not in self.supported_formats:
            print(f"❌ 不支持的格式: {output_format}")
            print(f"支持的格式: {', '.join(self.supported_formats)}")
            return False
        
        print(f"📝 LaTeX文件输出工具")
        print(f"输入文件: {tex_file}")
        print(f"输出格式: {output_format}")
        print("=" * 50)
        
        if output_format == 'pdf':
            return self.compile_to_pdf(tex_file, **kwargs)
        elif output_format == 'html':
            return self.convert_to_html(tex_file, output_file)
        elif output_format == 'txt':
            return self.convert_to_text(tex_file, output_file)
        elif output_format == 'md':
            return self.convert_to_markdown(tex_file, output_file)
        
        return False

def main():
    parser = argparse.ArgumentParser(description='LaTeX文件输出工具')
    parser.add_argument('tex_file', help='LaTeX文件路径')
    parser.add_argument('-f', '--format', choices=['html', 'txt', 'md', 'pdf'], 
                       default='html', help='输出格式 (默认: html)')
    parser.add_argument('-o', '--output', help='输出文件路径')
    parser.add_argument('--compiler', choices=['pdflatex', 'xelatex', 'lualatex'], 
                       default='pdflatex', help='PDF编译器 (默认: pdflatex)')
    parser.add_argument('--output-dir', help='PDF输出目录')
    
    args = parser.parse_args()
    
    tool = LaTeXOutputTool()
    
    kwargs = {}
    if args.format == 'pdf':
        kwargs['compiler'] = args.compiler
        if args.output_dir:
            kwargs['output_dir'] = args.output_dir
    
    success = tool.process_file(args.tex_file, args.format, args.output, **kwargs)
    
    if success:
        print("\n✅ 处理完成!")
    else:
        print("\n❌ 处理失败!")
        sys.exit(1)

if __name__ == '__main__':
    # 如果没有命令行参数，使用默认设置
    if len(sys.argv) == 1:
        tool = LaTeXOutputTool()
        tex_file = "interference_fringe_analysis_paper.tex"
        
        if not Path(tex_file).exists():
            print(f"❌ 错误: 找不到文件 {tex_file}")
            sys.exit(1)
        
        print("📝 LaTeX文件输出工具")
        print("=" * 50)
        print("🔍 检查LaTeX编译器...")
        
        available_compilers = tool.check_latex_installation()
        if available_compilers:
            print(f"✅ 找到编译器: {', '.join(available_compilers)}")
            print("🔄 尝试编译PDF...")
            if tool.compile_to_pdf(tex_file):
                print("✅ PDF编译成功!")
            else:
                print("❌ PDF编译失败，生成HTML预览...")
                tool.convert_to_html(tex_file)
        else:
            print("❌ 未找到LaTeX编译器")
            print("🔄 生成HTML预览...")
            tool.convert_to_html(tex_file)
            print("🔄 生成文本版本...")
            tool.convert_to_text(tex_file)
            print("🔄 生成Markdown版本...")
            tool.convert_to_markdown(tex_file)
        
        print("\n✅ 处理完成!")
    else:
        main()