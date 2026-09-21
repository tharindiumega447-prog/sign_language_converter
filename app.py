"""
Flask Application for Sign Language Alphabet (A-Z) to Text Conversion
Royal Blue, Crisp White & Green Edition
"""

import os
from flask import Flask, render_template

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

@app.route('/')
def index():
    """Renders the Sign Language Alphabet (A-Z) Converter interface."""
    return render_template('index.html')

if __name__ == '__main__':
    print("=" * 60)
    print("  Flask Sign Language Alphabet (A-Z) to Text Server")
    print("  Theme: Royal Blue, Crisp White & Green Edition")
    print("  Server running on http://127.0.0.1:5000 / http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)


