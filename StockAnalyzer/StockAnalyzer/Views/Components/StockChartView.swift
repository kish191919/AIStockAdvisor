// Views/Components/StockChartView.swift
// Views/Components/StockChartView.swift
import SwiftUI
import Charts

struct StockChartView: View {
    let chartData: ChartData
    @State private var selectedTimeframe: Timeframe = .daily
    @State private var showingIndicators = false
    
    enum Timeframe {
        case daily
        case monthly
    }
    
    var body: some View {
        VStack(spacing: 16) {
            // Timeframe Selector
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
            Group {
                if selectedTimeframe == .daily {
                    DailyChartView(data: chartData.dailyData, showingIndicators: showingIndicators)
                } else {
                    MonthlyChartView(data: chartData.monthlyData, showingIndicators: showingIndicators)
                }
            }
            .frame(height: 300)
            
            // Technical Indicators Panel
            if showingIndicators {
                ScrollView(.horizontal, showsIndicators: false) {
                    if selectedTimeframe == .daily {
                        if let lastDaily = chartData.dailyData.last {
                            TechnicalIndicatorsView(data: lastDaily)
                                .padding(.horizontal)
                        }
                    } else {
                        if let lastMonthly = chartData.monthlyData.last {
                            MonthlyIndicatorsView(data: lastMonthly)
                                .padding(.horizontal)
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemBackground))
        .cornerRadius(10)
        .shadow(radius: 2)
    }
}

// MARK: - Daily Chart View
struct DailyChartView: View {
    let data: [DailyData]
    let showingIndicators: Bool
    
    var body: some View {
        Chart {
            ForEach(data) { item in
                // Candlestick Body
                RectangleMark(
                    x: .value("Date", item.date),
                    yStart: .value("Open", item.open),
                    yEnd: .value("Close", item.close),
                    width: 4
                )
                .foregroundStyle(item.close > item.open ? Color.green.opacity(0.8) : Color.red.opacity(0.8))
                
                // High-Low Line
                RuleMark(
                    x: .value("Date", item.date),
                    yStart: .value("High", item.high),
                    yEnd: .value("Low", item.low)
                )
                .foregroundStyle(item.close > item.open ? Color.green.opacity(0.8) : Color.red.opacity(0.8))
                
                if showingIndicators {
                    // Bollinger Bands
                    if let upper = item.upperBand {
                        LineMark(
                            x: .value("Date", item.date),
                            y: .value("Upper Band", upper)
                        )
                        .foregroundStyle(.gray.opacity(0.5))
                    }
                    
                    if let lower = item.lowerBand {
                        LineMark(
                            x: .value("Date", item.date),
                            y: .value("Lower Band", lower)
                        )
                        .foregroundStyle(.gray.opacity(0.5))
                    }
                    
                    if let sma = item.sma {
                        LineMark(
                            x: .value("Date", item.date),
                            y: .value("SMA", sma)
                        )
                        .foregroundStyle(.blue.opacity(0.5))
                    }
                }
            }
        }
        .chartXAxis {
            AxisMarks(position: .bottom, values: .automatic(desiredCount: 5))
        }
        .chartYAxis {
            AxisMarks(position: .leading, values: .automatic(desiredCount: 5))
        }
    }
}

// MARK: - Monthly Chart View
struct MonthlyChartView: View {
    let data: [MonthlyData]
    let showingIndicators: Bool
    
    var body: some View {
        Chart {
            ForEach(data) { item in
                // Candlestick Body
                RectangleMark(
                    x: .value("Date", item.date),
                    yStart: .value("Open", item.open),
                    yEnd: .value("Close", item.close),
                    width: 4
                )
                .foregroundStyle(item.close > item.open ? Color.green.opacity(0.8) : Color.red.opacity(0.8))
                
                // High-Low Line
                RuleMark(
                    x: .value("Date", item.date),
                    yStart: .value("High", item.high),
                    yEnd: .value("Low", item.low)
                )
                .foregroundStyle(item.close > item.open ? Color.green.opacity(0.8) : Color.red.opacity(0.8))
                
                if showingIndicators {
                    if let ma20 = item.ma20 {
                        LineMark(
                            x: .value("Date", item.date),
                            y: .value("MA20", ma20)
                        )
                        .foregroundStyle(.blue.opacity(0.5))
                    }
                    
                    if let ma60 = item.ma60 {
                        LineMark(
                            x: .value("Date", item.date),
                            y: .value("MA60", ma60)
                        )
                        .foregroundStyle(.purple.opacity(0.5))
                    }
                }
            }
        }
        .chartXAxis {
            AxisMarks(position: .bottom, values: .automatic(desiredCount: 5))
        }
        .chartYAxis {
            AxisMarks(position: .leading, values: .automatic(desiredCount: 5))
        }
    }
}

// MARK: - Technical Indicators Views
struct TechnicalIndicatorsView: View {
    let data: DailyData
    
    var body: some View {
        HStack(spacing: 20) {
            if let rsi = data.rsi {
                VStack(alignment: .leading) {
                    Text("RSI")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.2f", rsi))
                        .foregroundColor(rsiColor(rsi))
                }
            }
            
            if let macd = data.macd {
                VStack(alignment: .leading) {
                    Text("MACD")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.2f", macd))
                }
            }
            
            if let sma = data.sma {
                VStack(alignment: .leading) {
                    Text("SMA")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.2f", sma))
                }
            }
        }
    }
    
    private func rsiColor(_ value: Double) -> Color {
        if value >= 70 { return .red }
        if value <= 30 { return .green }
        return .primary
    }
}

struct MonthlyIndicatorsView: View {
    let data: MonthlyData
    
    var body: some View {
        HStack(spacing: 20) {
            if let ma20 = data.ma20 {
                VStack(alignment: .leading) {
                    Text("MA20")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.2f", ma20))
                }
            }
            
            if let ma60 = data.ma60 {
                VStack(alignment: .leading) {
                    Text("MA60")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(String(format: "%.2f", ma60))
                }
            }
        }
    }
}

#Preview {
    StockChartView(chartData: ChartData(
        dailyData: [
            DailyData(date: "2024-01-01", open: 100, high: 105, low: 98, close: 103, volume: 1000000,
                     rsi: 65, macd: 2.5, signalLine: 2.0, macdHistogram: 0.5,
                     upperBand: 106, lowerBand: 97, sma: 101)
        ],
        monthlyData: [
            MonthlyData(date: "2024-01", open: 100, high: 110, low: 95, close: 108,
                       volume: 5000000, ma10: 105, ma20: 103, ma60: 100, ma120: 98)
        ]
    ))
    .padding()
}
