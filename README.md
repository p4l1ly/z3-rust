High level interface to the Z3 SMT solver. Currently, only support for boolean
logic is implemented. It is therefore usable only as a SAT solver. We have not
yet considered thread safety or reasonable behaviour if multiple contexts are
used at once.

```
use z3::Stage;

// Start to use Z3.
let mut ctx = z3::Context::new();

// Create a variable.
let foo = ctx.var_from_string("foo");

// Add it to the solver.
ctx.assert(foo);

{
    // Push Z3 solver's stack.
    let mut p = ctx.push();

    // A basic example of combining formulae.
    let foo_and_not_foo = p.and(vec![foo.inherit(), p.not(foo)]);
    p.assert(foo_and_not_foo);

    // No way to satisfy foo && !foo.
    assert!(!p.is_sat())

} // Pop Z3 solver's stack.

assert!(ctx.is_sat())
```
