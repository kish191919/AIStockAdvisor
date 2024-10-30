import SwiftUI

struct InteractiveChartOverlay: View {
    let data: [YahooChartDataPoint]
    @Binding var selectedPoint: YahooChartDataPoint?
    @State private var dragLocation: CGPoint = .zero
    
    var body: some View {
        GeometryReader { geometry in
            Color.clear
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            updateSelection(at: value.location, in: geometry.size)
                        }
                        .onEnded { _ in
                            // 선택 유지
                        }
                )
                .onChange(of: dragLocation) { _ in
                    updateSelection(at: dragLocation, in: geometry.size)
                }
        }
    }
    
    private func updateSelection(at point: CGPoint, in size: CGSize) {
        guard !data.isEmpty else { return }
        
        let xPosition = point.x
        let percentage = xPosition / size.width
        let index = Int((Double(data.count - 1) * percentage).rounded())
        let safeIndex = max(0, min(index, data.count - 1))
        selectedPoint = data[safeIndex]
    }
}
