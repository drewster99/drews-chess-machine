import SwiftUI

extension MarkdownText {
    /// One parsed markdown block rendered as a SwiftUI view.
    /// Routes by block kind to the right styling. Inline markdown
    /// inside a block (bold, italic, `code`, links) is delegated to
    /// `AttributedString(markdown:)`; on parser rejection we fall
    /// back to plain text rather than dropping content.
    struct BlockView: View {
        let block: Block

        var body: some View {
            switch block {
            case .heading(let level, let text):
                Text(Self.inline(text))
                    .font(.system(size: Self.headingSize(level), weight: .bold))
                    .padding(.top, level <= 2 ? 8 : 4)
                    .padding(.bottom, 2)
            case .paragraph(let text):
                Text(Self.inline(text))
                    .fixedSize(horizontal: false, vertical: true)
            case .list(let items, let ordered):
                VStack(alignment: .leading, spacing: 3) {
                    ForEach(Array(items.enumerated()), id: \.offset) { idx, item in
                        HStack(alignment: .top, spacing: 6) {
                            Text(ordered ? "\(idx + 1)." : "•")
                                .monospacedDigit()
                                .frame(width: 22, alignment: .trailing)
                            Text(Self.inline(item))
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }
            case .codeBlock(let code):
                Text(code)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(8)
                    .background(Color.secondary.opacity(0.15))
                    .cornerRadius(4)
            }
        }

        /// Inline-only Markdown parse for a single block of text.
        /// `.inlineOnlyPreservingWhitespace` keeps literal spacing in
        /// the caller's string (important for multi-line paragraphs)
        /// while still rendering `**bold**`, `*italic*`,
        /// `` `code` ``, and `[links](url)`. On parser rejection
        /// (unbalanced delimiters, unusual escape sequences) we fall
        /// through to a plain-text `AttributedString` rather than
        /// dropping the text — the reader should always see what
        /// Claude produced, even if it's not styled.
        private static func inline(_ text: String) -> AttributedString {
            let options = AttributedString.MarkdownParsingOptions(
                interpretedSyntax: .inlineOnlyPreservingWhitespace
            )
            do {
                return try AttributedString(markdown: text, options: options)
            } catch {
                return AttributedString(text)
            }
        }

        private static func headingSize(_ level: Int) -> CGFloat {
            switch level {
            case 1: return 22
            case 2: return 19
            case 3: return 16
            case 4: return 14
            default: return 13
            }
        }
    }
}
