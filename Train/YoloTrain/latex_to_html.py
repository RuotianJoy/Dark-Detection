#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaTeX转HTML工具
当没有LaTeX编译器时，将LaTeX文件转换为HTML格式进行预览
"""

import re
import os
from pathlib import Path

def latex_to_html(tex_file, output_file=None):
    """
    将LaTeX文件转换为HTML格式
    
    Args:
        tex_file (str): LaTeX文件路径
        output_file (str): 输出HTML文件路径
    """
    if not Path(tex_file).exists():
        print(f"❌ 错误: 文件 {tex_file} 不存在")
        return False
    
    # 读取LaTeX文件
    with open(tex_file, 'r', encoding='utf-8') as f:
        latex_content = f.read()
    
    # 转换为HTML
    html_content = convert_latex_to_html(latex_content)
    
    # 确定输出文件名
    if output_file is None:
        output_file = Path(tex_file).stem + '.html'
    
    # 写入HTML文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ 转换完成: {output_file}")
    return True

def convert_latex_to_html(latex_content):
    """
    将LaTeX内容转换为HTML
    """
    # 基础HTML模板 - 使用字符串拼接避免格式化问题
    html_head = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LaTeX论文预览</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }
        };
    </script>
    <style>
        body { font-family: 'Times New Roman', serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; background-color: #f9f9f9; }
        .paper { background: white; padding: 40px; box-shadow: 0 0 10px rgba(0,0,0,0.1); border-radius: 5px; }
        h1 { text-align: center; font-size: 1.5em; margin-bottom: 10px; }
        .author { text-align: center; margin-bottom: 20px; font-style: italic; }
        .abstract { background: #f0f0f0; padding: 15px; margin: 20px 0; border-left: 4px solid #007acc; }
        .keywords { font-weight: bold; margin-top: 10px; }
        h2 { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 5px; }
        h3 { color: #555; }
        .equation { text-align: center; margin: 20px 0; }
        .table { margin: 20px 0; border-collapse: collapse; width: 100%; }
        .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        .table th { background-color: #f2f2f2; }
        .itemize { margin: 10px 0; }
        .enumerate { margin: 10px 0; }
        .bibliography { margin-top: 30px; }
        .bibitem { margin: 5px 0; }
        .note { background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="paper">'''
    
    html_foot = '''    </div>
</body>
</html>'''
    
    # 提取文档内容（去掉导言区）
    content_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
    if content_match:
        content = content_match.group(1)
    else:
        content = latex_content
    
    # 提取标题
    title_match = re.search(r'\\title\{(.*?)\}', latex_content, re.DOTALL)
    if title_match:
        title = title_match.group(1)
        content = f"<h1>{clean_latex_text(title)}</h1>\n" + content
    
    # 提取作者
    author_match = re.search(r'\\author\{(.*?)\}', latex_content, re.DOTALL)
    if author_match:
        author = author_match.group(1)
        content = content.replace('\\maketitle', f'<div class="author">{clean_latex_text(author)}</div>')
    else:
        content = content.replace('\\maketitle', '')
    
    # 转换各种LaTeX命令
    content = convert_latex_commands(content)
    
    return html_head + content + html_foot

def convert_latex_commands(content):
    """
    转换LaTeX命令为HTML
    """
    # 处理摘要
    content = re.sub(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', 
                    r'<div class="abstract"><strong>摘要</strong><br>\1</div>', 
                    content, flags=re.DOTALL)
    
    # 处理章节标题
    content = re.sub(r'\\section\*?\{(.*?)\}', r'<h2>\1</h2>', content)
    content = re.sub(r'\\subsection\*?\{(.*?)\}', r'<h3>\1</h3>', content)
    content = re.sub(r'\\subsubsection\*?\{(.*?)\}', r'<h4>\1</h4>', content)
    
    # 处理文本格式
    content = re.sub(r'\\textbf\{(.*?)\}', r'<strong>\1</strong>', content)
    content = re.sub(r'\\textit\{(.*?)\}', r'<em>\1</em>', content)
    content = re.sub(r'\\emph\{(.*?)\}', r'<em>\1</em>', content)
    
    # 处理列表
    content = re.sub(r'\\begin\{itemize\}(.*?)\\end\{itemize\}', 
                    convert_itemize, content, flags=re.DOTALL)
    content = re.sub(r'\\begin\{enumerate\}(.*?)\\end\{enumerate\}', 
                    convert_enumerate, content, flags=re.DOTALL)
    
    # 处理数学公式
    content = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', 
                    r'<div class="equation">$$\1$$</div>', content, flags=re.DOTALL)
    
    # 处理表格
    content = re.sub(r'\\begin\{table\}.*?\\begin\{tabular\}\{.*?\}(.*?)\\end\{tabular\}.*?\\end\{table\}', 
                    convert_table, content, flags=re.DOTALL)
    
    # 处理参考文献
    content = re.sub(r'\\begin\{thebibliography\}\{.*?\}(.*?)\\end\{thebibliography\}', 
                    convert_bibliography, content, flags=re.DOTALL)
    
    # 处理引用
    content = re.sub(r'\\cite\{.*?\}', '[引用]', content)
    content = re.sub(r'\\bibitem\{.*?\}', '', content)
    
    # 清理其他LaTeX命令
    content = clean_latex_text(content)
    
    # 处理段落
    content = re.sub(r'\n\s*\n', '</p><p>', content)
    content = '<p>' + content + '</p>'
    content = content.replace('<p></p>', '')
    
    return content

def convert_itemize(match):
    """转换无序列表"""
    items = match.group(1)
    items = re.sub(r'\\item\s*', '<li>', items)
    items = re.sub(r'\n\s*<li>', '</li>\n<li>', items)
    return f'<ul class="itemize">{items}</li></ul>'

def convert_enumerate(match):
    """转换有序列表"""
    items = match.group(1)
    items = re.sub(r'\\item\s*', '<li>', items)
    items = re.sub(r'\n\s*<li>', '</li>\n<li>', items)
    return f'<ol class="enumerate">{items}</li></ol>'

def convert_table(match):
    """转换表格"""
    table_content = match.group(1)
    rows = table_content.split('\\\\')
    html_rows = []
    
    for i, row in enumerate(rows):
        if row.strip():
            cells = row.split('&')
            if i == 0:  # 表头
                html_cells = [f'<th>{clean_latex_text(cell.strip())}</th>' for cell in cells]
                html_rows.append(f'<tr>{"".join(html_cells)}</tr>')
            else:
                html_cells = [f'<td>{clean_latex_text(cell.strip())}</td>' for cell in cells]
                html_rows.append(f'<tr>{"".join(html_cells)}</tr>')
    
    return f'<table class="table">{"".join(html_rows)}</table>'

def convert_bibliography(match):
    """转换参考文献"""
    bib_content = match.group(1)
    return f'<div class="bibliography"><h2>参考文献</h2>{bib_content}</div>'

def clean_latex_text(text):
    """清理LaTeX文本中的特殊命令"""
    # 移除常见的LaTeX命令
    text = re.sub(r'\\[a-zA-Z]+\*?\{.*?\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+\*?', '', text)
    text = re.sub(r'\{|\}', '', text)
    text = re.sub(r'\\\\', '<br>', text)
    text = re.sub(r'\\&', '&', text)
    text = re.sub(r'\\%', '%', text)
    text = re.sub(r'\\\$', '$', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def main():
    print("📝 LaTeX转HTML工具")
    print("=" * 50)
    
    tex_file = "interference_fringe_analysis_paper.tex"
    
    if not Path(tex_file).exists():
        print(f"❌ 错误: 找不到文件 {tex_file}")
        return
    
    print(f"🔄 开始转换: {tex_file}")
    
    success = latex_to_html(tex_file)
    
    if success:
        html_file = Path(tex_file).stem + '.html'
        print(f"\n✅ 转换完成!")
        print(f"📄 HTML文件: {html_file}")
        print(f"💡 用浏览器打开 {html_file} 查看论文预览")
        print("\n📝 注意: 这是简化的HTML预览版本")
        print("📚 如需完整PDF版本，请安装LaTeX编译器")
        
        # 尝试在浏览器中打开
        try:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(html_file)}')
            print("🌐 已在浏览器中打开预览")
        except:
            pass
    else:
        print("\n❌ 转换失败")

if __name__ == '__main__':
    main()