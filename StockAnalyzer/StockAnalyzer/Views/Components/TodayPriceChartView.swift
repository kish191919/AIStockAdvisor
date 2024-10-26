// Views/Components/TodayPriceChartView.swift

import SwiftUI
import Charts
import Combine

struct TodayPriceChartView: View {
    let dailyData: [DailyData]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Today's Price Trend")
                .font(.headline)
            
            // 디버깅 정보 추가
            VStack(alignment: .leading) {
                Text("Debug Info:")
                    .font(.caption)
                    .foregroundColor(.gray)
                Text("Data count: \(dailyData.count)")
                    .font(.caption)
                if let firstData = dailyData.first {
                    Text("First data - Date: \(firstData.date), Close: \(firstData.close)")
                        .font(.caption)
                }
                if let lastData = dailyData.last {
                    Text("Last data - Date: \(lastData.date), Close: \(lastData.close)")
                        .font(.caption)
                }
            }
            .padding(.bottom)
            
            if dailyData.isEmpty {
                Text("No data available")
                    .foregroundColor(.secondary)
            } else {
                Chart {
                    ForEach(Array(dailyData.suffix(20)), id: \.id) { data in
                        LineMark(
                            x: .value("Date", data.date),
                            y: .value("Close", data.close)
                        )
                        .foregroundStyle(Color.blue)
                        
                        AreaMark(
                            x: .value("Date", data.date),
                            y: .value("Close", data.close)
                        )
                        .foregroundStyle(Color.blue.opacity(0.1))
                    }
                }
                .chartXAxis {
                    AxisMarks(position: .bottom) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel {
                            if let date = value.as(String.self) {
                                Text(formatDate(date))
                                    .font(.caption)
                            }
                        }
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel {
                            if let price = value.as(Double.self) {
                                Text("$\(String(format: "%.2f", price))")
                                    .font(.caption)
                            }
                        }
                    }
                }
            }
        }
    }
    
    // 날짜 포맷 함수
    private func formatDate(_ dateString: String) -> String {
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
        
        if let date = dateFormatter.date(from: dateString) {
            dateFormatter.dateFormat = "MM/dd"
            return dateFormatter.string(from: date)
        }
        return dateString
    }
}
