# How to apply WikiJS files to your own WikiJS instance

1. Go to the WikiJS admin panel
2. Go to 'Settings'
3. Click on 'Theme'
4. On the right side, see the "Code Injection" section
5. Copy the `head.html` and `body.html` files to the 'Head HTML Injection' and 'Body HTML Injection' fields, respectively
6. Change the `const CHATBOT_API_URL` to the URL of your DeSi instance (inside the 'Head HTML Injection' field)
7. Copy the `css_override.css` file to the CSS Override field
8. Click on 'Apply'
9. In the 'app.py', change the `allow_origins=` variable in `app.add_middleware(` to the URL of the used WikiJS instance (currently test URL)
10. With the DeSi chatbot deployed in `const CHATBOT_API_URL`, check in the 'Home' DataStore WikiJS page, that the DeSi chatbot button appears and you can communicate with it.