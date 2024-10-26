// Views/Components/StockChartView.swift
// Views/Components/StockChartView.swift
import SwiftUI
import Charts
import Combine


struct StockChartView: View {
    let chartData: ChartData
    @State private var selectedTimeframe: Timeframe = .daily
    @State private var showingIndicators = false
    
    enum Timeframe {
        case daily
        case monthly
    }
    
    var currentData: [any ChartDataPoint] {
        selectedTimeframe == .daily ? chartData.dailyData : chartData.monthlyData
    }
    
    var body: some View {
        VStack(spacing: 16) {
            // Debug Info
            VStack(alignment: .leading, spacing: 4) {
                Text("Debug Information")
                    .font(.headline)
                Text("Daily data count: \(chartData.dailyData.count)")
                Text("Monthly data count: \(chartData.monthlyData.count)")
                if let firstDaily = chartData.dailyData.first {
                    Text("First daily data:")
                    Text("Date: \(firstDaily.date)")
                    Text("Open: \(firstDaily.open)")
                    Text("Close: \(firstDaily.close)")
                }
            }
            .font(.caption)
            .padding()
            .background(Color.gray.opacity(0.1))
            .cornerRadius(8)
            
            // Controls
            HStack {
                Picker("Timeframe", selection: $selectedTimeframe) {
                    Text("Daily").tag(Timeframe.daily)
                    Text("Monthly").tag(Timeframe.monthly)
                }
                .pickerStyle(SegmentedPickerStyle())
                
                Toggle("Indicators", isOn: $showingIndicators)
            }
            .padding(.horizontal)
            
            // Chart
            if currentData.isEmpty {
                Text("No data available")
                    .foregroundColor(.secondary)
            } else {
                Chart {
                    ForEach(Array(currentData.enumerated()), id: \.1.date) { index, item in
                        RectangleMark(
                            x: .value("Date", index), // 인덱스를 사용하여 x축 값 설정
                            yStart: .value("Price", min(item.open, item.close)),
                            yEnd: .value("Price", max(item.open, item.close)),
                            width: 6
                        )
                        .foregroundStyle(item.close > item.open ? Color.green : Color.red)
                        
                        RuleMark(
                            x: .value("Date", index),
                            yStart: .value("Price", item.low),
                            yEnd: .value("Price", item.high)
                        )
                        .foregroundStyle(item.close > item.open ? Color.green : Color.red)
                    }
                }
                .frame(height: 300)
                .chartXAxis {
                    AxisMarks(position: .bottom) { value in
                        if let index = value.as(Int.self),
                           index < currentData.count {
                            AxisGridLine()
                            AxisTick()
                            AxisValueLabel {
                                Text(currentData[index].date)
                                    .font(.caption)
                            }
                        }
                    }
                }
                .chartYAxis {
                    AxisMarks(position: .leading) { value in
                        AxisGridLine()
                        AxisTick()
                        AxisValueLabel()
                    }
                }
            }
            
            // Indicators
            if showingIndicators {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 20) {
                        if selectedTimeframe == .daily {
                            if let lastDaily = chartData.dailyData.last {
                                DailyIndicators(data: lastDaily)
                            }
                        } else {
                            if let lastMonthly = chartData.monthlyData.last {
                                MonthlyIndicators(data: lastMonthly)
                            }
                        }
                    }
                    .padding(.horizontal)
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
        .onAppear {
            print("StockChartView appeared with data:")
            print("Daily data count: \(chartData.dailyData.count)")
            print("Monthly data count: \(chartData.monthlyData.count)")
        }
    }
}

// MARK: - Protocols
protocol ChartDataPoint {
    var date: String { get }
    var open: Double { get }
    var high: Double { get }
    var low: Double { get }
    var close: Double { get }
}

extension DailyData: ChartDataPoint {}
extension MonthlyData: ChartDataPoint {}

// MARK: - Indicator Views
struct DailyIndicators: View {
    let data: DailyData
    
    var body: some View {
        HStack(spacing: 20) {
            if let rsi = data.rsi {
                IndicatorItem(title: "RSI", value: rsi)
            }
            if let macd = data.macd {
                IndicatorItem(title: "MACD", value: macd)
            }
            if let sma = data.sma {
                IndicatorItem(title: "SMA", value: sma)
            }
        }
    }
}

struct MonthlyIndicators: View {
    let data: MonthlyData
    
    var body: some View {
        HStack(spacing: 20) {
            if let ma20 = data.ma20 {
                IndicatorItem(title: "MA20", value: ma20)
            }
            if let ma60 = data.ma60 {
                IndicatorItem(title: "MA60", value: ma60)
            }
        }
    }
}

struct IndicatorItem: View {
    let title: String
    let value: Double
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            Text(String(format: "%.2f", value))
                .font(.subheadline)
        }
    }
}

// MARK: - Preview
struct StockChartView_Previews: PreviewProvider {
    static var previews: some View {
        StockChartView(chartData: ChartData(
            dailyData: [
                DailyData(date: "2024-01-01", open: 100, high: 105, low: 98, close: 103,
                         volume: 1000000, rsi: 65, macd: 2.5, signalLine: 2.0,
                         macdHistogram: 0.5, upperBand: 106, lowerBand: 97, sma: 101)
            ],
            monthlyData: [
                MonthlyData(date: "2024-01", open: 100, high: 110, low: 95, close: 108,
                           volume: 5000000, ma10: 105, ma20: 103, ma60: 100, ma120: 98)
            ]
        ))
        .padding()
    }
}
