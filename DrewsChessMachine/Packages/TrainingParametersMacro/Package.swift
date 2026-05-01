// swift-tools-version: 6.0
import PackageDescription
import CompilerPluginSupport

let package = Package(
    name: "TrainingParametersMacro",
    platforms: [.macOS(.v15)],
    products: [
        .library(
            name: "TrainingParametersMacroSupport",
            targets: ["TrainingParametersMacroSupport"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/swiftlang/swift-syntax.git", from: "600.0.0")
    ],
    targets: [
        .macro(
            name: "TrainingParametersMacroPlugin",
            dependencies: [
                .product(name: "SwiftSyntax", package: "swift-syntax"),
                .product(name: "SwiftSyntaxMacros", package: "swift-syntax"),
                .product(name: "SwiftCompilerPlugin", package: "swift-syntax")
            ]
        ),
        .target(
            name: "TrainingParametersMacroSupport",
            dependencies: ["TrainingParametersMacroPlugin"]
        ),
        .testTarget(
            name: "TrainingParametersMacroTests",
            dependencies: [
                "TrainingParametersMacroPlugin",
                .product(name: "SwiftSyntaxMacrosTestSupport", package: "swift-syntax")
            ]
        )
    ]
)
