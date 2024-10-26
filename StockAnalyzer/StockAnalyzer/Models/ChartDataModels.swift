

// Models/ChartDataModels.swift
import SwiftUI
import Charts
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요
import Foundation
import Foundation

struct ChartData: Codable {
    let dailyData: [DailyData]
    let monthlyData: [MonthlyData]
    
    enum CodingKeys: String, CodingKey {
        case dailyData = "daily_data"
        case monthlyData = "monthly_data"
    }
}

struct DailyData: Codable, Identifiable {
    let id = UUID()
    let date: String
    let open: Double
    let high: Double
    let low: Double
    let close: Double
    let volume: Int
    let rsi: Double?
    let macd: Double?
    let signalLine: Double?
    let macdHistogram: Double?
    let upperBand: Double?
    let lowerBand: Double?
    let sma: Double?
    
    enum CodingKeys: String, CodingKey {
        case date
        case open
        case high
        case low
        case close
        case volume
        case rsi = "RSI"
        case macd = "MACD"
        case signalLine = "Signal_Line"
        case macdHistogram = "MACD_Histogram"
        case upperBand = "Upper_Band"
        case lowerBand = "Lower_Band"
        case sma = "SMA"
    }
}

struct MonthlyData: Codable, Identifiable {
    let id = UUID()
    let date: String
    let open: Double
    let high: Double
    let low: Double
    let close: Double
    let volume: Int
    let ma10: Double?
    let ma20: Double?
    let ma60: Double?
    let ma120: Double?
    
    enum CodingKeys: String, CodingKey {
        case date
        case open
        case high
        case low
        case close
        case volume
        case ma10 = "MA_10"
        case ma20 = "MA_20"
        case ma60 = "MA_60"
        case ma120 = "MA_120"
    }
}
// NewsData 모델 수정
struct NewsItem: Codable, Identifiable {
    var id: String { title }
    let title: String
    let date: String?
    let published_at: String?
    let pubDate: String?
    let url: String?
    
    var displayDate: String {
        return date ?? published_at ?? pubDate ?? "No date"
    }
}
