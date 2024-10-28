
//Views/Components/ResultView.swift
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요
import Charts  // 추가


import SwiftUI
import Charts  // Charts import 추가

struct ResultView: View {
    let result: StockAnalysisResponse
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            // Today's Price Chart
            if let chartData = result.chartData {
                TodayPriceChartView(dailyData: chartData.dailyData)
                    .frame(height: 200)
                    .padding()
                    .background(Color(.systemBackground))
                    .cornerRadius(10)
                    .shadow(radius: 2)
            }
            
            // Decision Section
            DecisionSection(result: result)
            
            // Price Section
            PriceSection(result: result)
            
            // Market Indicators Section
            MarketIndicatorsSection(result: result)
        }
        .padding()
    }
}
