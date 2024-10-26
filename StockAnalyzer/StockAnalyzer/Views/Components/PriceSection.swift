//Views/Components/PriceSection.swift
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

// Views/Components/PriceSection.swift 수정
struct PriceSection: View {
    let result: StockAnalysisResponse
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Price Information")
                .font(.headline)
            
            HStack {
                VStack(alignment: .leading) {
                    Text("Current Price")
                        .font(.subheadline)
                    Text("$\(String(format: "%.2f", result.currentPrice))")
                        .font(.title2)
                }
                
                Spacer()
                
                VStack(alignment: .trailing) {
                    Text("Expected Next Day")
                        .font(.subheadline)
                    Text("$\(String(format: "%.2f", result.expectedNextDayPrice))")
                        .font(.title2)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
    }
}
