import SwiftUI
import Charts
internal import Combine


/*
 As of Xcode 26.4, macOS 26.4, these are true:

 1. Scrolling with a large amount of data is FAST

 2. ZOOMing the chart with a large amount of data iS SUPER SLOW

 3. Appending ONE SINGLE ELEMENT when there is a large amount of data is VERY SLOW

 */
struct DataPoint: Identifiable {
    let id: Int
    let x: Double
    let y: Double
}

enum ChartType: String, CaseIterable, Identifiable {
    case native = "Native Chart"
    case manual = "Manual Path"
    var id: String { self.rawValue }
}

struct ContentView: View {
    // Generate 100,000 data points
    @State var data: [DataPoint] = (0..<100_000).map { i in
        DataPoint(id: i, x: Double(i), y: sin(Double(i) * 0.01) + Double.random(in: -0.1...0.1))
    }
    @State var foo = 0
    @State private var selectedChart: ChartType = .native
    let timer = Timer.publish(
        every: 0.01,
        on: .main,
        in: .default
    ).autoconnect()


    var body: some View {
        VStack {
            Text("Points: \(data.count)")
            HStack {
                Text("SwiftUI Performance Test (100,000 points)")
                    .font(.headline)
                Spacer()
                Picker("View Mode", selection: $selectedChart) {
                    ForEach(ChartType.allCases) { type in
                        Text(type.rawValue).tag(type)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 300)
            }
            .padding()

            Group {
                switch selectedChart {
                case .native:
                    ChartView(data: data)
                case .manual:
                    PathChartView(data: data)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding()
        }
        .frame(minWidth: 800, minHeight: 600)
        .onReceive(timer) { _ in
            let dp = DataPoint(id: 100000 + foo,
                               x: Double(100000 + foo),
                               y: cos(Double(foo) * 0.03)
            )
            foo += 1
            data.append(dp)
        }
    }
}

struct ChartView: View {
    let data: [DataPoint]
    @State private var zoomLevel: CGFloat = 1.0
    @GestureState private var magnifyBy = 1.0

    var body: some View {
        Chart(data) { point in
            LineMark(
                x: .value("X", point.x),
                y: .value("Y", point.y)
            )
            .interpolationMethod(.linear)
        }
        .chartXScale(domain: 0...(100_000 / (zoomLevel * magnifyBy)))
        .gesture(
            MagnificationGesture()
                .updating($magnifyBy) { value, state, _ in
                    state = value
                }
                .onEnded { value in
                    zoomLevel *= value
                }
        )
        .chartScrollableAxes(.horizontal)
    }
}

struct PathChartView: View {
    let data: [DataPoint]
    @State private var zoomLevel: CGFloat = 1.0
    @GestureState private var magnifyBy = 1.0

    var body: some View {
        ScrollView(.horizontal) {
            ChartLineShape(data: data)
                .stroke(
                    LinearGradient(
                        gradient: Gradient(colors: [.blue, .purple]),
                        startPoint: .top,
                        endPoint: .bottom
                    ),
                    style: StrokeStyle(lineWidth: 1)
                )
                .drawingGroup()
                .containerRelativeFrame(
                    .horizontal,
                    alignment: .leading,
                    { width, _ in
                        width * zoomLevel * magnifyBy
                    }
                )
        }
        .gesture(
            MagnificationGesture()
                .updating($magnifyBy) { value, state, _ in
                    state = value
                }
                .onEnded { value in
                    zoomLevel = max(1.0, zoomLevel * value)
                }
        )
        .border(Color.secondary)
    }
}

struct ChartLineShape: Shape {
    let data: [DataPoint]

    func path(in rect: CGRect) -> Path {
        var path = Path()
        guard data.count > 1 else { return path }

        let step = rect.width / CGFloat(data.count - 1)
        var lastX: CGFloat = .leastNonzeroMagnitude
        for (index, point) in data.enumerated() {
            // Normalize y from -1.2...1.2 to 0...1
            let val = (point.y + 1.2) / 2.4
            let point = CGPoint(
                x: CGFloat(index) * step,
                y: (1 - CGFloat(val)) * rect.height
            )

            if index == 0 {
                path.move(to: point)
            } else {
                // Note: I added this check to improve performance but I'm not sure it did
                if lastX != point.x {
                    path.addLine(to: point)
                    lastX = point.x
                }
            }
        }
        return path
    }
}

#Preview {
    ContentView()
}
