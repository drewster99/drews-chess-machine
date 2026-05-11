import SwiftUI
import Charts

struct DataPoint: Identifiable {
    let id: Int
    let x: Double
    let y: Double
}

struct ContentView: View {
    // Generate 100,000 data points
    let data: [DataPoint] = (0..<100_000).map { i in
        DataPoint(id: i, x: Double(i), y: sin(Double(i) * 0.01) + Double.random(in: -0.1...0.1))
    }
    
    @State private var scrollPosition: Double = 0
    
    var body: some View {
        VStack {
            Text("SwiftUI Charts Performance Test (100,000 points)")
                .font(.headline)
                .padding()
            
            Chart(data) { point in
                LineMark(
                    x: .value("X", point.x),
                    y: .value("Y", point.y)
                )
                .interpolationMethod(.linear)
            }
            .chartXScale(domain: 0...100_000)
            // Enable native horizontal scrolling
            .chartScrollableAxis(.horizontal)
            .chartXVisibleDomain(length: 1000) // Show 1000 points at a time
            .chartScrollPosition(x: $scrollPosition)
            .frame(height: 400)
            .padding()
            
            HStack {
                Text("Scroll Position: \(Int(scrollPosition))")
                Spacer()
                Button("Jump to Start") { scrollPosition = 0 }
                Button("Jump to Middle") { scrollPosition = 50000 }
                Button("Jump to End") { scrollPosition = 99000 }
            }
            .padding()
        }
        .frame(minWidth: 800, minHeight: 600)
    }
}

#Preview {
    ContentView()
}
