// YahooFinanceChartView.swift

import SwiftUI
import Combine
import Charts

// ViewModel and Data Models
public struct YahooChartDataPoint: Identifiable {  // 이름 변경
    public let id = UUID()
    public let timestamp: Date
    public let open: Double
    public let high: Double
    public let low: Double
    public let close: Double
    public let volume: Int
}

public class YahooChartViewModel: ObservableObject {
    @Published public var chartData: [ChartPeriod: [YahooChartDataPoint]] = [:]  // 타입 변경
    @Published public var isLoading = false
    @Published public var error: String?
    @Published public var currentSymbol: String = ""
    
    public init() {}
    
    public func fetchChartData(symbol: String, period: ChartPeriod) {
        isLoading = true
        error = nil
        currentSymbol = symbol
        
        let baseUrl = "https://query1.finance.yahoo.com/v8/finance/chart/"
        let queryParams = "interval=\(period.interval)&range=\(period.range)"
        
        guard let url = URL(string: "\(baseUrl)\(symbol)?\(queryParams)") else {
            error = "Invalid URL"
            isLoading = false
            return
        }
        
        URLSession.shared.dataTask(with: url) { [weak self] data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    self?.error = error.localizedDescription
                    self?.isLoading = false
                    return
                }
                
                guard let data = data else {
                    self?.error = "No data received"
                    self?.isLoading = false
                    return
                }
                
                // ViewModel 내의 fetchChartData 메서드에서 아래 부분을 수정
                do {
                    let response = try JSONDecoder().decode(YahooChartResponse.self, from: data)
                    if let chartError = response.chart.error {
                        self?.error = chartError.description
                        self?.isLoading = false
                        return
                    }
                    
                    guard let result = response.chart.result?.first,
                          let timestamps = result.timestamp,
                          let quote = result.indicators.quote.first else {
                        self?.error = "Invalid data format"
                        self?.isLoading = false
                        return
                    }
                    
                    var chartPoints: [YahooChartDataPoint] = []
                    
                    for i in 0..<timestamps.count {
                        // 수정된 부분: 옵셔널 체이닝 및 기본값 처리
                        let open = quote.open?[i] ?? quote.close?[i] ?? 0.0
                        let high = quote.high?[i] ?? quote.close?[i] ?? 0.0
                        let low = quote.low?[i] ?? quote.close?[i] ?? 0.0
                        let close = quote.close?[i] ?? quote.open?[i] ?? 0.0
                        let volume = quote.volume?[i] ?? 0
                        
                        let point = YahooChartDataPoint(
                            timestamp: Date(timeIntervalSince1970: TimeInterval(timestamps[i])),
                            open: open,
                            high: high,
                            low: low,
                            close: close,
                            volume: volume
                        )
                        chartPoints.append(point)
                    }
                    
                    self?.chartData[period] = chartPoints
                    self?.isLoading = false
                } catch {
                    self?.error = "Failed to decode data: \(error.localizedDescription)"
                    self?.isLoading = false
                }
            }
        }.resume()
    }
}

struct YahooFinanceChartView: View {
    @StateObject private var viewModel = YahooChartViewModel()
    @State private var selectedPeriod: ChartPeriod = .oneDay
    @State private var selectedPoint: YahooChartDataPoint?
    @State private var tooltipPosition: CGFloat = 0
    let symbol: String
    
    init(symbol: String) {
        self.symbol = symbol
    }
    
    private func filterValidData(_ data: [YahooChartDataPoint]) -> [YahooChartDataPoint] {
        return data.filter { $0.close > 0 }
    }
    
    private func calculateYAxisRange(data: [YahooChartDataPoint]) -> ClosedRange<Double> {
        let validData = filterValidData(data)
        guard !validData.isEmpty else { return 0...100 }
        
        let values = validData.map { $0.close }
        let minPrice = values.min() ?? 0
        let maxPrice = values.max() ?? 100
        let padding = (maxPrice - minPrice) * 0.05
        
        return (minPrice - padding)...(maxPrice + padding)
    }
    
    private func formatDateTime(_ date: Date, period: ChartPeriod) -> String {
        let formatter = DateFormatter()
        
        switch period {
        case .oneDay, .fiveDay:
            formatter.dateFormat = "MM/dd HH:mm"
        case .oneMonth, .sixMonth, .yearToDate:
            formatter.dateFormat = "MM/dd"
        default:
            formatter.dateFormat = "yyyy/MM/dd"
        }
        
        return formatter.string(from: date)
    }
    
    private func formatPrice(_ price: Double) -> String {
        return String(format: "%.2f", price)
    }
    
    private var displayPrice: String {
        if let point = selectedPoint {
            return formatPrice(point.close)
        } else if let lastPrice = viewModel.chartData[selectedPeriod]?.last?.close {
            return formatPrice(lastPrice)
        }
        return "0.00"
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            // Stock Info Header
            Text(symbol.uppercased())
                .font(.title)
                .fontWeight(.bold)
                .padding(.horizontal)
            
            Text("$\(displayPrice)")
                .font(.title2)
                .foregroundColor(.secondary)
                .padding(.horizontal)
                .padding(.bottom, 8)
                .animation(.easeOut(duration: 0.2), value: displayPrice)
            
            // Chart Container with Bottom Divider
            VStack(spacing: 0) {
                ZStack(alignment: .topLeading) {
                    if viewModel.isLoading {
                        ProgressView()
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else if let error = viewModel.error {
                        Text(error)
                            .foregroundColor(.red)
                            .multilineTextAlignment(.center)
                            .padding()
                    } else if let data = viewModel.chartData[selectedPeriod] {
                        let validData = filterValidData(data)
                        
                        if validData.isEmpty {
                            Text("No valid data available")
                                .foregroundColor(.secondary)
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                        } else {
                            ZStack(alignment: .topLeading) {
                                Chart {
                                    ForEach(validData) { point in
                                        LineMark(
                                            x: .value("Time", point.timestamp),
                                            y: .value("Price", point.close)
                                        )
                                        .foregroundStyle(Color.blue)
                                        
                                        AreaMark(
                                            x: .value("Time", point.timestamp),
                                            y: .value("Price", point.close)
                                        )
                                        .foregroundStyle(
                                            LinearGradient(
                                                colors: [.blue.opacity(0.3), .clear],
                                                startPoint: .top,
                                                endPoint: .bottom
                                            )
                                        )
                                    }
                                    
                                    if let selectedPoint = selectedPoint {
                                        RuleMark(
                                            x: .value("Time", selectedPoint.timestamp)
                                        )
                                        .foregroundStyle(Color.gray.opacity(0.5))
                                        
                                        PointMark(
                                            x: .value("Time", selectedPoint.timestamp),
                                            y: .value("Price", selectedPoint.close)
                                        )
                                        .foregroundStyle(.blue)
                                        .symbolSize(100)
                                    }
                                }
                                .chartYScale(domain: calculateYAxisRange(data: validData))
                                .chartXAxis(.hidden)
                                .chartYAxis {
                                    AxisMarks(position: .leading) { value in
                                        if let price = value.as(Double.self) {
                                            AxisValueLabel {
                                                Text(formatPrice(price))
                                                    .font(.caption)
                                            }
                                        }
                                    }
                                }
                                
                                // Tooltip overlay
                                if let point = selectedPoint {
                                    Text("\(formatDateTime(point.timestamp, period: selectedPeriod))\n$\(formatPrice(point.close))")
                                        .font(.caption)
                                        .foregroundColor(.white)
                                        .padding(6)
                                        .background(Color.black.opacity(0.8))
                                        .cornerRadius(6)
                                        .offset(x: max(0, min(tooltipPosition - 40, UIScreen.main.bounds.width - 100)))
                                        .offset(y: 20)
                                }
                                
                                // Gesture overlay
                                GeometryReader { geometry in
                                    Rectangle()
                                        .fill(.clear)
                                        .contentShape(Rectangle())
                                        .gesture(
                                            DragGesture(minimumDistance: 0)
                                                .onChanged { value in
                                                    let currentX = value.location.x
                                                    tooltipPosition = currentX
                                                    
                                                    guard currentX >= 0,
                                                          currentX <= geometry.size.width
                                                    else { return }
                                                    
                                                    let startDate = validData.first?.timestamp ?? Date()
                                                    let endDate = validData.last?.timestamp ?? Date()
                                                    let timeRange = endDate.timeIntervalSince(startDate)
                                                    let xRatio = currentX / geometry.size.width
                                                    let targetDate = startDate.addingTimeInterval(timeRange * xRatio)
                                                    
                                                    selectedPoint = validData.min(by: {
                                                        abs($0.timestamp.timeIntervalSince(targetDate)) < abs($1.timestamp.timeIntervalSince(targetDate))
                                                    })
                                                }
                                                .onEnded { _ in
                                                    // Optional: Uncomment to clear selection when touch ends
                                                    // selectedPoint = nil
                                                    // tooltipPosition = 0
                                                }
                                        )
                                }
                            }
                        }
                    } else {
                        Text("No data available")
                            .foregroundColor(.secondary)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
                .frame(height: 250)
                .padding(.bottom, 16)
                .clipped() // 그래프의 내용이 밖으로 나가지 않도록 제한
                
                // Gradient Divider
                LinearGradient(
                    gradient: Gradient(colors: [
                        Color.gray.opacity(0.2),
                        Color.gray.opacity(0.1),
                        Color.gray.opacity(0.05)
                    ]),
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .frame(height: 1)
                .padding(.horizontal)
                
                // Period Selection Buttons
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(ChartPeriod.allCases, id: \.self) { period in
                            Button(action: {
                                selectedPeriod = period
                                selectedPoint = nil
                                if viewModel.chartData[period] == nil {
                                    viewModel.fetchChartData(symbol: symbol, period: period)
                                }
                            }) {
                                Text(period.rawValue)
                                    .font(.footnote)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(
                                        selectedPeriod == period ?
                                            Color.blue :
                                            Color.gray.opacity(0.2)
                                    )
                                    .foregroundColor(
                                        selectedPeriod == period ?
                                            .white :
                                            .primary
                                    )
                                    .cornerRadius(6)
                            }
                        }
                    }
                    .padding(.horizontal)
                    .padding(.vertical, 12)
                }
            }
            .background(Color(.systemBackground))
        }
        .onAppear {
            if viewModel.chartData[selectedPeriod] == nil {
                viewModel.fetchChartData(symbol: symbol, period: selectedPeriod)
            }
        }
    }
}

// Chart Period Enum
public enum ChartPeriod: String, CaseIterable {
    case oneDay = "1D"
    case fiveDay = "5D"
    case oneMonth = "1M"
    case sixMonth = "6M"
    case yearToDate = "YTD"
    case oneYear = "1Y"
    case twoYear = "2Y"
    case fiveYear = "5Y"
    case max = "MAX"
    
    public var interval: String {
        switch self {
        case .oneDay: return "2m"
        case .fiveDay: return "15m"
        case .oneMonth: return "30m"
        case .sixMonth: return "1d"
        case .yearToDate: return "1d"
        case .oneYear: return "1d"
        case .twoYear: return "1wk"
        case .fiveYear: return "1wk"
        case .max: return "1mo"
        }
    }
    
    public var range: String {
        switch self {
        case .oneDay: return "1d"
        case .fiveDay: return "5d"
        case .oneMonth: return "1mo"
        case .sixMonth: return "6mo"
        case .yearToDate: return "ytd"
        case .oneYear: return "1y"
        case .twoYear: return "2y"
        case .fiveYear: return "5y"
        case .max: return "max"
        }
    }
}

// Response Models
struct YahooChartResponse: Codable {
    let chart: YahooChart
}

struct YahooChart: Codable {
    let result: [YahooChartResult]?
    let error: YahooError?
}

struct YahooError: Codable {
    let code: String
    let description: String
}

struct YahooChartResult: Codable {
    let meta: YahooMeta
    let timestamp: [Int]?
    let indicators: YahooIndicators
}

struct YahooMeta: Codable {
    let currency: String
    let symbol: String
    let regularMarketPrice: Double?
    let previousClose: Double?
}

struct YahooIndicators: Codable {
    let quote: [YahooQuote]
}

struct YahooQuote: Codable {
    let high: [Double?]?
    let low: [Double?]?
    let open: [Double?]?
    let close: [Double?]?
    let volume: [Int?]?
}
