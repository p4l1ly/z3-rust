use std::ffi::{CString};
use std::marker::PhantomData;
use std::mem;

mod z {
    #![allow(dead_code)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// High level interface to the Z3 SMT solver. Currently, only support for
/// boolean logic is implemented. It is therefore usable only as a SAT solver.
/// We have not yet considered thread safety or reasonable behaviour if multiple
/// contexts are used at once.
///
/// # Examples
///
/// ```
/// use z3::Stage;
///
/// // Create Z3 config, context and solver.
/// let mut ctx = z3::Context::new();
///
/// // Create variable, which is of type Ast, which itself represents an
/// // immutable reference to a Z3 AST object.
/// let foo = ctx.var_from_string("foo");
///
/// ctx.assert(foo);
///
/// {
///     // Push Z3 solver's stack.
///     let mut p = ctx.push();
///
///     // A basic example of combining Asts.
///     let foo_and_not_foo = p.and(vec![foo.inherit(), p.not(foo)]);
///     p.assert(foo_and_not_foo);
///
///     // No way to satisfy foo && !foo.
///     assert!(!p.is_sat())
/// }
/// // Pop of the Z3 solver's stack happens here, with the drop of the push
/// // object p. Asts created between push and pop are no more valid, but this
/// // library ensures that the borrow checker would refuse any leak.
///
/// assert!(ctx.is_sat())
/// ```
///
/// The following gets refused by the borrow checker because the value
/// `not_foo`, which is created in the lifetime of `p` is used after `p` is
/// dropped.
/// ```compile_fail
/// use z3::Stage;
///
/// let mut ctx = z3::Context::new();
/// let foo = ctx.var_from_string("foo");
/// let not_foo;
///
/// {
///     let mut p = ctx.push();
///
///     not_foo = p.not(foo);
/// }
///
/// ctx.assert(not_foo);
/// ```
///
/// Tricks like this will also not work (because `ctx` is mutably borrowed by
/// `p`).
/// ```compile_fail
/// use z3::Stage;
///
/// let mut ctx = z3::Context::new();
/// let foo = ctx.var_from_string("foo");
/// let not_foo;
///
/// {
///     let mut p = ctx.push();
///
///     not_foo = ctx.not(foo);
/// }
///
/// ctx.assert(not_foo);
/// ```
#[derive(Debug)]
pub struct Context {
    ctx: z::Z3_context,
    cfg: z::Z3_config,
    sort: z::Z3_sort,
    ztrue: z::Z3_ast,
    zfalse: z::Z3_ast,
    solver: z::Z3_solver,
}

macro_rules! ast { ($ptr:expr) => { Ast{ptr: $ptr, phantom: PhantomData} } }

impl Context {
    pub fn new() -> Self {
        unsafe {
            let cfg = z::Z3_mk_config();
            let ctx = z::Z3_mk_context(cfg);
            let sort = z::Z3_mk_bool_sort(ctx);

            let solver = z::Z3_mk_solver(ctx);
            z::Z3_solver_inc_ref(ctx, solver);

            let ztrue = z::Z3_mk_true(ctx);
            let zfalse = z::Z3_mk_false(ctx);

            Context{ctx, cfg, sort, ztrue, zfalse, solver}
        }
    }

    pub fn ztrue<'a>(&self) -> Ast<'a, Context> {
        ast!(self.ztrue)
    }

    pub fn zfalse<'a>(&self) -> Ast<'a, Context> {
        ast!(self.zfalse)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            z::Z3_solver_dec_ref(self.ctx, self.solver);
            z::Z3_del_context(self.ctx);
            z::Z3_del_config(self.cfg);
        }
    }
}

/// An abstract object that gets created after calling Z3 solver push, and on
/// drop it calls Z3 solver pop. It cannot live longer than its parent (which is
/// Context or another Push object), and makes sure that no `Ast` created
/// between the push and pop can be used after the pop.
#[derive(Debug)]
pub struct Push<'a, T: 'a> {
    ctx: z::Z3_context,
    sort: z::Z3_sort,
    solver: z::Z3_solver,
    phantom: PhantomData<&'a T>,
}

impl<'a, T> Drop for Push<'a, T> {
    fn drop(&mut self) {
        unsafe { z::Z3_solver_pop(self.ctx, self.solver, 1); }
    }
}

/// Ast (= abstract syntax tree) is a pointer to a Z3 formula. It can be created
/// by methods of `Push` or `Context` object and should not live longer than the
/// object from which it was created.
#[derive(Debug)]
pub struct Ast<'a, T: 'a> {
    ptr: z::Z3_ast,
    phantom: PhantomData<&'a T>,
}

impl<'b, 'a: 'b, T> Ast<'a, T> {
    #[inline]
    pub fn inherit<T2: 'b>(self: &Ast<'a, T>) -> Ast<'b, T2> {
        ast!(self.ptr)
    }
}

impl<'a, T> Copy for Ast<'a, T> {}

impl<'a, T> Clone for Ast<'a, T> {
    #[inline] fn clone(&self) -> Ast<'a, T> { ast!(self.ptr) }
}

/// `Context` and `Push` have very similar semantics, both are references to the
/// Z3 context + config + solver, `Push` has only an extra functionality to pop
/// on drop. This module implements private common functions of `Context` and
/// `Push`.
mod ctx_like {
    use super::*;

    pub trait CtxLike {
        fn ctx(&self) -> z::Z3_context;
        fn sort(&self) -> z::Z3_sort;
        fn solver(&self) -> z::Z3_solver;
    }

    impl CtxLike for Context {
        #[inline] fn ctx(&self) -> z::Z3_context { self.ctx }
        #[inline] fn sort(&self) -> z::Z3_sort { self.sort }
        #[inline] fn solver(&self) -> z::Z3_solver { self.solver }
    }

    impl<'a, T> CtxLike for Push<'a, T> {
        #[inline] fn ctx(&self) -> z::Z3_context { self.ctx }
        #[inline] fn sort(&self) -> z::Z3_sort { self.sort }
        #[inline] fn solver(&self) -> z::Z3_solver { self.solver }
    }
}

/// `Context` and `Push` have very similar semantics, both are references to the
/// Z3 context + config + solver, `Push` has only an extra functionality to pop
/// on drop. This module implements public common functions of `Context` and
/// `Push`: operators of the logic, solver's assert, is_sat, and push functions.
pub trait Stage: ctx_like::CtxLike + Sized {
    fn push<'a>(&'a mut self) -> Push<'a, Self> {
        unsafe { z::Z3_solver_push(self.ctx(), self.solver()); }
        Push{
            ctx: self.ctx(),
            sort: self.sort(),
            solver: self.solver(),
            phantom: PhantomData,
        }
    }

    fn assert<'a, T>(&'a mut self, ast: Ast<'a, T>) {
        unsafe { z::Z3_solver_assert(self.ctx(), self.solver(), ast.ptr); }
    }

    fn is_sat(&mut self) -> bool {
        unsafe { match z::Z3_solver_check(self.ctx(), self.solver()) {
            z::Z3_lbool_Z3_L_TRUE => true,
            z::Z3_lbool_Z3_L_FALSE => false,
            _ => panic!("undefined solver check result"),
        }}
    }

    fn var_from_string<'a>(&self, name: &str) -> Ast<'a, Self> {
        unsafe {
            let symbol = z::Z3_mk_string_symbol(self.ctx(),
                CString::new(name).unwrap().as_ptr());
            ast!(z::Z3_mk_const(self.ctx(), symbol, self.sort()))
        }
    }

    fn not<'a, 'b: 'a, T>(&self, x: Ast<'b, T>) -> Ast<'a, Self> {
        unsafe { ast!(z::Z3_mk_not(self.ctx(), x.ptr)) }
    }

    fn eq<'a, 'b: 'a, 'c: 'a, T1, T2>(
        &self,
        a: Ast<'b, T1>,
        b: Ast<'c, T2>,
    ) -> Ast<'a, Self> {
        unsafe { ast!(z::Z3_mk_eq(self.ctx(), a.ptr, b.ptr)) }
    }

    fn ite<'a, 'b: 'a, 'c: 'a, 'd: 'a, T1, T2, T3>(
        &self,
        condition: Ast<'b, T1>,
        if_true: Ast<'c, T2>,
        if_false: Ast<'d, T3>,
    ) -> Ast<'a, Self> {
        unsafe { ast!(z::Z3_mk_ite(self.ctx(),
            condition.ptr, if_true.ptr, if_false.ptr)) }
    }

    fn iff<'a, 'b: 'a, 'c: 'a, T1, T2>(
        &self,
        condition: Ast<'b, T1>,
        value: Ast<'c, T2>
    ) -> Ast<'a, Self> {
        unsafe { ast!(z::Z3_mk_iff(self.ctx(), condition.ptr, value.ptr)) }
    }

    fn implies<'a, 'b: 'a, 'c: 'a, T1, T2>(
        &self,
        condition: Ast<'b, T1>,
        value: Ast<'c, T2>
    ) -> Ast<'a, Self> {
        unsafe { ast!(z::Z3_mk_implies(self.ctx(), condition.ptr, value.ptr)) }
    }

    fn xor<'a, 'b: 'a, 'c: 'a, T1, T2>(
        &self,
        a: Ast<'b, T1>,
        b: Ast<'c, T2>
    ) -> Ast<'a, Self> {
        unsafe { ast!(z::Z3_mk_xor(self.ctx(), a.ptr, b.ptr)) }
    }

    fn and<'a, 'b: 'a, T>(
        &self,
        operands: Vec<Ast<'b, T>>
    ) -> Ast<'a, Self> {
        unsafe { ast!(z::Z3_mk_and(self.ctx(),
            operands.len() as u32, mem::transmute(operands.as_ptr())
        )) }
    }

    fn or<'a, 'b: 'a, T>(
        &self,
        operands: Vec<Ast<'b, T>>
    ) -> Ast<'a, Self> {
        unsafe { ast!(z::Z3_mk_or(self.ctx(),
            operands.len() as u32,
            mem::transmute(operands.as_ptr())
        )) }
    }
}

impl Stage for Context {}
impl<'a, T> Stage for Push<'a, T> {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn double_push() {
        let mut ctx = Context::new();
        let foo = ctx.var_from_string("foo");
        let bar = ctx.var_from_string("bar");

        ctx.assert(foo);

        {
            let mut push1 = ctx.push();
            let baz = push1.var_from_string("baz");

            push1.assert(bar);
            {
                let mut push2 = push1.push();

                assert!(push2.is_sat());

                let nbar = push2.not(bar);
                push2.assert(nbar);

                assert!(!push2.is_sat());
            }

            assert!(push1.is_sat());

            let x = push1.and(vec![baz, push1.not(baz), bar.inherit()]);
            push1.assert(x);

            assert!(!push1.is_sat());
        }

        assert!(ctx.is_sat());
    }

    #[test]
    fn operators_coverage() {
        let mut ctx = Context::new();

        let x = ctx.or(vec![ctx.ztrue(), ctx.zfalse()]);
        let y = ctx.ite(ctx.eq(ctx.not(ctx.and(vec![x, x])), x), x, x);
        let z = ctx.xor(ctx.implies(ctx.iff(y, y), y), y);

        ctx.assert(z);
        println!("ctx.is_sat: {}", ctx.is_sat());
    }
}
