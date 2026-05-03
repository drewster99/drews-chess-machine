import SwiftUI

/// Lightweight block-level Markdown view. Supports headings (`#`..
/// `######`), unordered lists (`- ` / `* `), ordered lists
/// (`1. `), fenced code blocks (```` ``` ````), and paragraphs —
/// which is roughly the surface area of a typical `claude -p`
/// response. Inline formatting (bold, italic, `code`, links) is
/// delegated to `AttributedString(markdown:)`, which SwiftUI's
/// `Text` renders natively. Block types outside this list
/// (tables, blockquotes, horizontal rules) fall through as
/// paragraphs with their raw markers visible — acceptable for
/// a debug panel.
struct MarkdownText: View {
    let content: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Array(Block.parse(content).enumerated()), id: \.offset) { _, block in
                BlockView(block: block)
            }
        }
    }
}
