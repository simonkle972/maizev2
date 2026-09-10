// Shared message rendering for Maize chat surfaces.
//
// Used by the student chat (templates/chat.html) and the professor test chat
// (templates/professor/test_ta.html). Both must render LLM output identically —
// the professor is previewing what students will see — so this lives in one
// file rather than being copied into each template.
//
// Requires KaTeX auto-render (renderMathInElement) and DOMPurify; both are
// feature-detected, so a page that omits them degrades instead of throwing.

function preprocessLatex(text) {
    let processed = text;
    
    processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, '$$$$$1$$$$');
    processed = processed.replace(/\\\(([\s\S]*?)\\\)/g, '$$$1$$');
    
    return processed;
}

function sanitizeForRendering(text) {
    // Fix common LLM formatting issues that break rendering
    let sanitized = text;
    
    // 1. Remove asterisks that are adjacent to $ (math delimiters)
    //    e.g., "*$x$*" -> "$x$", "**$equation$**" -> "$equation$"
    sanitized = sanitized.replace(/\*+(\$[^\$]+\$)\*+/g, '$1');
    sanitized = sanitized.replace(/\*+(\$\$[\s\S]*?\$\$)\*+/g, '$1');
    
    // 2. Remove asterisks before LaTeX commands (e.g., "**\varepsilon" -> "\varepsilon")
    //    Must capture full command name (varepsilon, frac, cdot, etc.)
    sanitized = sanitized.replace(/\*+(\\[a-zA-Z]+)/g, '$1');
    sanitized = sanitized.replace(/(\\[a-zA-Z]+)\*+/g, '$1');
    
    // 3. Remove trailing asterisks at end of sentences/lines
    sanitized = sanitized.replace(/\s\*+\s*$/gm, '');
    sanitized = sanitized.replace(/\.\s*\*+\s/g, '. ');
    sanitized = sanitized.replace(/\.\s*\*+$/gm, '.');
    
    // 4. Remove asterisks around section headers like "*2b) Title:**" or "**2a) Title:**"
    sanitized = sanitized.replace(/^\*+\s*(\d+[a-zA-Z]?\))/gm, '$1');
    sanitized = sanitized.replace(/\*+\s*(\d+[a-zA-Z]?\))/g, '$1');
    sanitized = sanitized.replace(/:\s*\*+\s*$/gm, ':');
    sanitized = sanitized.replace(/:\s*\*+\s*\n/g, ':\n');
    
    // 5. Fix unbalanced $ signs by escaping lone $ that aren't part of pairs
    const dollarMatches = sanitized.match(/\$/g);
    if (dollarMatches && dollarMatches.length % 2 === 1) {
        sanitized = sanitized.replace(/\$(?=\d+[,.]?\d*(?:[^$]|$))/g, '\\$');
    }
    
    // 6. Fix asterisks inside what looks like math content
    sanitized = sanitized.replace(/(\w)\*(\w+)\*(\w)/g, '$1\\*$2\\*$3');
    
    // 7. Clean up long broken bold patterns
    sanitized = sanitized.replace(/\*\*([^\*]{100,})\*\*/g, (match, content) => {
        if (!content.includes(' ')) {
            return content;
        }
        return match;
    });
    
    // 8. Same for single asterisk italics
    sanitized = sanitized.replace(/\*([^\*]{100,})\*/g, (match, content) => {
        if (!content.includes(' ')) {
            return content;
        }
        return match;
    });
    
    // 9. Remove any remaining lone asterisks that aren't valid markdown
    //    Simple approach: remove * surrounded by spaces or at line boundaries
    sanitized = sanitized.replace(/\s\*\s/g, ' ');
    sanitized = sanitized.replace(/^\*\s/gm, '');
    sanitized = sanitized.replace(/\s\*$/gm, '');
    
    // 10. Strip all asterisks if content has math (aggressive cleanup for math-heavy responses)
    if (sanitized.includes('$') || sanitized.includes('\\frac') || sanitized.includes('\\varepsilon')) {
        // Remove bold/italic markers entirely - they just cause problems with math
        // First remove bold (**text**)
        sanitized = sanitized.replace(/\*\*([^*]+)\*\*/g, '$1');
        // Then remove italic (*text*) - Safari-compatible without lookbehind
        // Match *text* where text doesn't start/end with * and has no newlines
        sanitized = sanitized.replace(/\*([^*\n]+)\*/g, '$1');
    }

    // 11. Escape # characters in LaTeX expressions
    // In LaTeX, # is a special character for macro parameters and must be escaped as \#
    // Process display math $$...$$ first to avoid matching $$ as two inline $
    sanitized = sanitized.replace(/\$\$([\s\S]+?)\$\$/g, function(match, content) {
        // Escape # that aren't already escaped (Safari-compatible)
        // Split by \#, then rejoin with \#, escaping any remaining #
        let escaped = content.split('\\#').map(function(part) {
            return part.replace(/#/g, '\\#');
        }).join('\\#');
        return '$$' + escaped + '$$';
    });
    // Process inline math $...$
    sanitized = sanitized.replace(/\$([^$]+?)\$/g, function(match, content) {
        // Escape # that aren't already escaped (Safari-compatible)
        let escaped = content.split('\\#').map(function(part) {
            return part.replace(/#/g, '\\#');
        }).join('\\#');
        return '$' + escaped + '$';
    });

    return sanitized;
}

function formatContent(content) {
    // First sanitize problematic patterns
    content = sanitizeForRendering(content);
    content = preprocessLatex(content);
    
    const mathPlaceholders = [];
    let placeholderIndex = 0;
    
    // Protect display math first ($$...$$)
    content = content.replace(/\$\$([\s\S]*?)\$\$/g, (match) => {
        mathPlaceholders.push(match);
        return `%%%MATH_DISPLAY_${placeholderIndex++}%%%`;
    });
    
    // Protect inline math ($...$) - improved regex to be more robust
    // Only match if there's content between $ signs and no newlines
    content = content.replace(/\$([^\$\n]+?)\$/g, (match, inner) => {
        // Skip if the inner content looks like it's not math (e.g., currency)
        if (/^\d+[,.]?\d*$/.test(inner.trim())) {
            return match; // Leave currency-like patterns alone
        }
        mathPlaceholders.push(match);
        return `%%%MATH_INLINE_${placeholderIndex++}%%%`;
    });
    
    // Apply markdown formatting
    content = content.replace(/\n/g, '<br>');
    
    // Bold - only match if content is reasonable length and has spaces
    content = content.replace(/\*\*(.{1,80}?)\*\*/g, '<strong>$1</strong>');
    
    // Italic - only match if content is reasonable length
    // Safari-compatible: match *text* but only if not inside ** markers
    // We already handled ** for bold above, so remaining single * pairs are italic
    content = content.replace(/\*([^\*\n]{1,80}?)\*/g, '<em>$1</em>');
    
    // Code
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Restore math placeholders
    for (let i = 0; i < mathPlaceholders.length; i++) {
        // Function replacements, not strings: the placeholders hold raw math, and
        // `$$` in a string replacement is the escape for a literal `$` — which
        // silently collapsed every $$display$$ block down to $inline$.
        content = content.replace(`%%%MATH_DISPLAY_${i}%%%`, () => mathPlaceholders[i]);
        content = content.replace(`%%%MATH_INLINE_${i}%%%`, () => mathPlaceholders[i]);
    }
    
    // Sanitize HTML to prevent XSS from LLM prompt injection
    content = `<p>${content}</p>`;
    if (typeof DOMPurify !== 'undefined') {
        content = DOMPurify.sanitize(content, {
            ALLOWED_TAGS: ['p', 'br', 'strong', 'b', 'em', 'i', 'code', 'pre', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote', 'a', 'span', 'div', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'hr', 'sup', 'sub'],
            ALLOWED_ATTR: ['href', 'target', 'rel', 'class']
        });
    }
    return content;
}

function renderLatex(element) {
    if (typeof renderMathInElement !== 'undefined') {
        try {
            renderMathInElement(element, {
                delimiters: [
                    {left: '$$', right: '$$', display: true},
                    {left: '$', right: '$', display: false},
                    {left: '\\[', right: '\\]', display: true},
                    {left: '\\(', right: '\\)', display: false}
                ],
                throwOnError: false,
                errorColor: '#cc0000',
                strict: false,
                trust: true,
                macros: {
                    "\\R": "\\mathbb{R}",
                    "\\N": "\\mathbb{N}",
                    "\\Z": "\\mathbb{Z}",
                    "\\Q": "\\mathbb{Q}",
                    "\\C": "\\mathbb{C}"
                },
                // Custom error handler - show clean fallback instead of ugly error
                errorCallback: function(msg, err) {
                    console.warn('KaTeX error:', msg, err);
                }
            });
            
            // Post-process: find any KaTeX error spans and clean them up
            const errorSpans = element.querySelectorAll('.katex-error');
            errorSpans.forEach(span => {
                // Extract the original text and display it cleanly
                const originalText = span.getAttribute('title') || span.textContent;
                // Remove the $ delimiters for cleaner display
                const cleanText = originalText.replace(/^\$+|\$+$/g, '');
                span.className = 'math-fallback';
                span.style.cssText = 'font-family: "Times New Roman", serif; font-style: italic; color: inherit;';
                span.textContent = cleanText;
            });
        } catch (e) {
            console.warn('LaTeX rendering error:', e);
        }
    }
}

