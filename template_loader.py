import os

def load_template(template_name):
    template_path = os.path.join(os.path.dirname(__file__), 'templates', template_name)
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"<!-- Template {template_name} not found -->"
    except Exception as e:
        return f"<!-- Error loading template {template_name}: {str(e)} -->"

def load_css(css_filename):
    css_path = os.path.join(os.path.dirname(__file__), 'static', css_filename)
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()
            return f"<style>\n{css_content}\n</style>"
    except FileNotFoundError:
        return f"<!-- CSS file {css_filename} not found -->"
    except Exception as e:
        return f"<!-- Error loading CSS {css_filename}: {str(e)} -->"
