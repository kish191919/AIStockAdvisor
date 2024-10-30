import SwiftUI

struct HistoryView: View {
    var body: some View {
        NavigationView {
            StockAnalysisHistoryView()
                .navigationTitle("History")
        }
    }
}
