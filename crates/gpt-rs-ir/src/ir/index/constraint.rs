//! Constraints: comparisons of two expressions, joined by `or`.

use std::fmt;

use anyhow::Result;
use smallvec::SmallVec;

use super::{AccessExpr, IndexEnv};

/// The relation of a comparison of two index expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Relation {
    Less,
    LessEqual,
    Equal,
    NotEqual,
    GreaterEqual,
    Greater,
}

impl Relation {
    pub(super) const ALL: [Relation; 6] = [
        Relation::Less,
        Relation::LessEqual,
        Relation::Equal,
        Relation::NotEqual,
        Relation::GreaterEqual,
        Relation::Greater,
    ];

    /// The symbol of the relation in index constraints.
    pub fn symbol(self) -> &'static str {
        match self {
            Relation::Less => "<",
            Relation::LessEqual => "<=",
            Relation::Equal => "==",
            Relation::NotEqual => "!=",
            Relation::GreaterEqual => ">=",
            Relation::Greater => ">",
        }
    }

    pub fn holds<T: PartialOrd>(self, lhs: T, rhs: T) -> bool {
        match self {
            Relation::Less => lhs < rhs,
            Relation::LessEqual => lhs <= rhs,
            Relation::Equal => lhs == rhs,
            Relation::NotEqual => lhs != rhs,
            Relation::GreaterEqual => lhs >= rhs,
            Relation::Greater => lhs > rhs,
        }
    }
}

/// A comparison of two index expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cmp {
    pub lhs: AccessExpr,
    pub op: Relation,
    pub rhs: AccessExpr,
}

impl Cmp {
    pub fn new(lhs: impl Into<AccessExpr>, op: Relation, rhs: impl Into<AccessExpr>) -> Self {
        Self {
            lhs: lhs.into(),
            op,
            rhs: rhs.into(),
        }
    }

    pub fn holds(&self, env: &dyn IndexEnv) -> Result<bool> {
        Ok(self.op.holds(self.lhs.eval(env)?, self.rhs.eval(env)?))
    }
}

/// A constraint of a `tile.block`: it holds when any of its comparisons does.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Constraint(pub SmallVec<[Cmp; 1]>);

impl Constraint {
    /// The constraint `lhs op rhs`.
    pub fn cmp(lhs: impl Into<AccessExpr>, op: Relation, rhs: impl Into<AccessExpr>) -> Self {
        let mut cmps = SmallVec::new();
        cmps.push(Cmp::new(lhs, op, rhs));
        Self(cmps)
    }

    pub fn holds(&self, env: &dyn IndexEnv) -> Result<bool> {
        for cmp in &self.0 {
            if cmp.holds(env)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn exprs(&self) -> impl Iterator<Item = &AccessExpr> {
        self.0.iter().flat_map(|cmp| [&cmp.lhs, &cmp.rhs])
    }

    pub fn map_exprs(&self, rewrite: &dyn Fn(&AccessExpr) -> AccessExpr) -> Constraint {
        let mut cmps = SmallVec::new();
        for cmp in &self.0 {
            cmps.push(Cmp::new(rewrite(&cmp.lhs), cmp.op, rewrite(&cmp.rhs)));
        }
        Constraint(cmps)
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for (position, cmp) in self.0.iter().enumerate() {
            if position > 0 {
                f.write_str(" or ")?;
            }
            write!(f, "{} {} {}", cmp.lhs, cmp.op.symbol(), cmp.rhs)?;
        }
        Ok(())
    }
}
