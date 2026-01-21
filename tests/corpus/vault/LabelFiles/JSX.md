# JSX

JSX is an XML-like syntax extension to ECMAScript without any defined semantics. It's NOT intended to be implemented by engines or browsers. **It's NOT a proposal to incorporate JSX into the ECMAScript spec itself.** It's intended to be used by various preprocessors (transpilers) to transform these tokens into standard ECMAScript.

<https://facebook.github.io/jsx/>

```
// Using JSX to express UI components
var dropdown =
  <Dropdown>
    A dropdown list
    <Menu>
      <MenuItem>Do Something</MenuItem>
      <MenuItem>Do Something Fun!</MenuItem>
      <MenuItem>Do Something Else</MenuItem>
    </Menu>
  </Dropdown>;

render(dropdown);

```

## With Typescript 

<https://www.typescriptlang.org/docs/handbook/jsx.html>

[JSX](https://facebook.github.io/jsx/) is an embeddable XML-like syntax. It is meant to be transformed into valid JavaScript, though the semantics of that transformation are implementation-specific. JSX rose to popularity with the [React](https://reactjs.org/) framework, but has since seen other implementations as well. TypeScript supports embedding, type checking, and compiling JSX directly to JavaScript