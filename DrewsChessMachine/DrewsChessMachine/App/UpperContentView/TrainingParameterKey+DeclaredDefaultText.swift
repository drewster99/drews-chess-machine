import Foundation

// MARK: - Declared-default text for settings-popover fields
//
// An empty edit field in the Training and Arena settings popovers shows its
// placeholder, and a placeholder reads as "this is the default". So it has to
// be the parameter's declared default, rendered in the same format the row
// writes its own value with (the model's `seedFromParams()` and the row's
// stepper binding) — otherwise a blank field shows a number that is neither
// what is stored nor what the parameter resets to.
//
// The popovers used to restate every default twice per row, as the
// placeholder string and as the stepper's parse-failure fallback. The
// declared defaults moved and most of those copies did not, so the UI
// advertised defaults the app no longer had. Rows now take both from the
// declaration: the typed value via `declaredDefault`, the text via these
// helpers. A row whose text is a unit conversion of the stored value (the
// autosave interval is edited in minutes, stored in seconds; the arena
// interval is a duration spec) converts the declared default through the same
// function that seeds the field instead of using these directly.

extension TrainingParameterKey where Value == Double {
    /// The declared default rendered with `format`, a `String(format:)`
    /// specifier. Pass the exact format the row uses for its own text.
    static func declaredDefaultText(format: String) -> String {
        String(format: format, declaredDefault)
    }
}

extension TrainingParameterKey where Value == Int {
    /// The declared default as the plain integer text the row writes back.
    static var declaredDefaultText: String {
        String(declaredDefault)
    }
}

extension TrainingParameterKey where Value == UInt64 {
    /// The declared default as the plain decimal text the row writes back.
    static var declaredDefaultText: String {
        String(declaredDefault)
    }
}
