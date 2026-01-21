# Webpack js

Webpack is a **bundler** that uses a dependency graph to bundle our code and assets (incuding static assets such as images) into a ‘bundle’ which we can then deploy.

Creating a huge monolithic JavaScript file is ugly and difficult to maintain, but multiple JavaScript files require multiple requests to fetch, and can add significant overhead. The solution is to write code splitting it into as many files as you need, and using `require()`(and as we’ll soon see, `import` ) statements to connect them together as we see fit. When we `require` something from a file, that becomes a dependency of that file. All our interconnected files and assets form a **dependency graph**. Then we use a bundler to to parse through these files and connect them appropriately based on the `require` and `import` statements and perform some more optimizations on top of it to generate one (sometimes more than one) JavaScript files which we can then use.

Webpack can also load non-js files, such as static assets, JSON files, CSS and more.

<https://medium.com/@agzuniverse/webpack-and-babel-what-are-they-and-how-to-use-them-with-react-5807afc82ca8>