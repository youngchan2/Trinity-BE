use thiserror::Error;

/// Byte offsets in a text input. Programmatically constructed nodes have no span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceSpan {
    pub start: usize,
    pub end: usize,
}

/// Small syntax adapter for the postprocessed S-expression dialect.
///
/// Expressions are retained without arithmetic rewrites. This is not a second
/// optimizer IR or an e-graph: callers can construct nodes directly, and the text
/// reader is useful for the existing Python backend's corpus.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrNode {
    head: String,
    args: Option<Vec<IrNode>>,
    span: Option<SourceSpan>,
}

impl IrNode {
    pub fn atom(value: impl Into<String>) -> Self {
        Self {
            head: value.into(),
            args: None,
            span: None,
        }
    }

    pub fn call(op: impl Into<String>, args: impl IntoIterator<Item = Self>) -> Self {
        Self {
            head: op.into(),
            args: Some(args.into_iter().collect()),
            span: None,
        }
    }

    pub fn head(&self) -> &str {
        &self.head
    }

    pub fn args(&self) -> Option<&[Self]> {
        self.args.as_deref()
    }

    pub fn span(&self) -> Option<SourceSpan> {
        self.span
    }

    pub fn parse(text: &str) -> Result<Self, ParseError> {
        let mut parser = Parser { text, pos: 0 };
        let node = parser.node(0)?;
        parser.whitespace();
        if parser.pos != text.len() {
            return Err(parser.error("unexpected trailing input"));
        }
        Ok(node)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("IR syntax at byte {offset}: {message}")]
pub struct ParseError {
    pub offset: usize,
    pub message: &'static str,
}

struct Parser<'a> {
    text: &'a str,
    pos: usize,
}

impl Parser<'_> {
    fn error(&self, message: &'static str) -> ParseError {
        ParseError {
            offset: self.pos,
            message,
        }
    }

    fn whitespace(&mut self) {
        loop {
            while self.pos < self.text.len() && self.text.as_bytes()[self.pos].is_ascii_whitespace()
            {
                self.pos += 1;
            }
            if self.text.as_bytes().get(self.pos) != Some(&b';') {
                break;
            }
            while self.pos < self.text.len() && self.text.as_bytes()[self.pos] != b'\n' {
                self.pos += 1;
            }
        }
    }

    fn atom(&mut self) -> Result<String, ParseError> {
        let start = self.pos;
        while self.pos < self.text.len() {
            let byte = self.text.as_bytes()[self.pos];
            if byte.is_ascii_whitespace() || matches!(byte, b'(' | b')' | b';') {
                break;
            }
            self.pos += 1;
        }
        if start == self.pos {
            return Err(self.error("expected an atom"));
        }
        Ok(self.text[start..self.pos].to_owned())
    }

    fn node(&mut self, depth: usize) -> Result<IrNode, ParseError> {
        self.whitespace();
        if depth >= 256 {
            return Err(self.error("nesting limit exceeded"));
        }
        let start = self.pos;
        let mut node = if self.text.as_bytes().get(self.pos) == Some(&b'(') {
            self.pos += 1;
            self.whitespace();
            let head = self.atom()?;
            let mut args = Vec::new();
            loop {
                self.whitespace();
                match self.text.as_bytes().get(self.pos) {
                    Some(b')') => {
                        self.pos += 1;
                        break;
                    }
                    None => return Err(self.error("unclosed expression")),
                    _ => args.push(self.node(depth + 1)?),
                }
            }
            IrNode::call(head, args)
        } else {
            IrNode::atom(self.atom()?)
        };
        node.span = Some(SourceSpan {
            start,
            end: self.pos,
        });
        Ok(node)
    }
}
