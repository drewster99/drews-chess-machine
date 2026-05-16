// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftUIFastCharts",
    platforms: [.macOS(.v15)],
    products: [
        .library(
            name: "SwiftUIFastCharts",
            targets: ["SwiftUIFastCharts"]
        )
    ],
    targets: [
        .target(name: "SwiftUIFastCharts"),
        .testTarget(
            name: "SwiftUIFastChartsTests",
            dependencies: ["SwiftUIFastCharts"]
        )
    ]
)
