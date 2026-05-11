// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "swiftui-charts-performance",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "PerformanceTest", targets: ["PerformanceTest"])
    ],
    targets: [
        .executableTarget(
            name: "PerformanceTest",
            path: "."
        )
    ]
)
