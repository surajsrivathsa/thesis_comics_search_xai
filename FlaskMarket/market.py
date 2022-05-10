from flask import Flask
app = Flask(__name__) # builtin variable name that refers to current file


# decorator - what url must this be displayed on web
@app.route('/')
def hello_world():
    return '<h1>Hello, World!!!</h1>'


@app.route('/about')
def about_page():
    return '<h1> About Page </h1>'


# pass names dynamically to webpage
@app.route('/about/<username>')
def about_page_username(username):
    return f'<h1> About Page : {username}</h1>'








