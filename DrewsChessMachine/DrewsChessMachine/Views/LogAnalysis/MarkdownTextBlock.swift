import Foundation

extension MarkdownText {
    enum Block {
        case heading(level: Int, text: String)
        case paragraph(String)
        case list(items: [String], ordered: Bool)
        case codeBlock(String)

        static func parse(_ content: String) -> [Block] {
            var blocks: [Block] = []
            let lines = content.components(separatedBy: "\n")
            var i = 0
            while i < lines.count {
                let line = lines[i]
                let trimmed = line.trimmingCharacters(in: .whitespaces)

                // Fenced code block — consume until the matching closing
                // fence. Unterminated fences swallow the rest of the
                // document, which matches GitHub-flavored markdown.
                if trimmed.hasPrefix("```") {
                    i += 1
                    var code = ""
                    while i < lines.count,
                          !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                        code += lines[i] + "\n"
                        i += 1
                    }
                    if i < lines.count { i += 1 } // consume closing fence
                    blocks.append(.codeBlock(
                        code.trimmingCharacters(in: CharacterSet.newlines)
                    ))
                    continue
                }

                // Blank line — paragraph separator.
                if trimmed.isEmpty {
                    i += 1
                    continue
                }

                // ATX heading (`#` through `######` followed by a space).
                if let level = headingLevel(trimmed: trimmed) {
                    let headerStart = trimmed.index(trimmed.startIndex, offsetBy: level + 1)
                    let headerText = String(trimmed[headerStart...])
                    blocks.append(.heading(level: level, text: headerText))
                    i += 1
                    continue
                }

                // Unordered list.
                if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") {
                    var items: [String] = []
                    while i < lines.count {
                        let t = lines[i].trimmingCharacters(in: .whitespaces)
                        if t.hasPrefix("- ") || t.hasPrefix("* ") {
                            items.append(String(t.dropFirst(2)))
                            i += 1
                        } else {
                            break
                        }
                    }
                    blocks.append(.list(items: items, ordered: false))
                    continue
                }

                // Ordered list (`N. text`).
                if orderedListItem(trimmed: trimmed) != nil {
                    var items: [String] = []
                    while i < lines.count {
                        let t = lines[i].trimmingCharacters(in: .whitespaces)
                        if let rest = orderedListItem(trimmed: t) {
                            items.append(rest)
                            i += 1
                        } else {
                            break
                        }
                    }
                    blocks.append(.list(items: items, ordered: true))
                    continue
                }

                // Default: paragraph. Collect consecutive non-blank,
                // non-special lines into a single paragraph string so
                // inline markdown can span multiple source lines.
                var para = line
                i += 1
                while i < lines.count {
                    let next = lines[i]
                    let nextTrim = next.trimmingCharacters(in: .whitespaces)
                    if nextTrim.isEmpty { break }
                    if nextTrim.hasPrefix("#") || nextTrim.hasPrefix("```")
                        || nextTrim.hasPrefix("- ") || nextTrim.hasPrefix("* ") {
                        break
                    }
                    if orderedListItem(trimmed: nextTrim) != nil { break }
                    para += "\n" + next
                    i += 1
                }
                blocks.append(.paragraph(para))
            }
            return blocks
        }

        /// ATX-style heading detector. Returns `1…6` if `trimmed`
        /// begins with that many `#` followed by a space; `nil`
        /// otherwise. Bare `#######` (7+) is not a heading per CommonMark.
        private static func headingLevel(trimmed: String) -> Int? {
            var count = 0
            for ch in trimmed {
                if ch == "#" {
                    count += 1
                    if count > 6 { return nil }
                } else if ch == " " {
                    return count > 0 ? count : nil
                } else {
                    return nil
                }
            }
            return nil // hashes-only line isn't a heading
        }

        /// Parse a single ordered-list item line (`N. text`). Returns
        /// the text after `N. ` or `nil` if the line isn't one.
        private static func orderedListItem(trimmed: String) -> String? {
            var idx = trimmed.startIndex
            var hasDigit = false
            while idx < trimmed.endIndex, trimmed[idx].isNumber {
                hasDigit = true
                idx = trimmed.index(after: idx)
            }
            guard hasDigit, idx < trimmed.endIndex, trimmed[idx] == "." else { return nil }
            let afterDot = trimmed.index(after: idx)
            guard afterDot < trimmed.endIndex, trimmed[afterDot] == " " else { return nil }
            let textStart = trimmed.index(after: afterDot)
            return String(trimmed[textStart...])
        }
    }
}
