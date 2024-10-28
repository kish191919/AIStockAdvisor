
import SwiftUI
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

// Models/StockAnalysisModels.swift 수정
struct StockAnalysisRequest: Encodable {
    let symbol: String
    let language: String
    
    enum CodingKeys: String, CodingKey {
        case symbol
        case language
    }
}

// MARK: - Response Model
struct StockAnalysisResponse: Decodable {
    let decision: String
    let percentage: Int
    let reason: String
    let currentPrice: Double
    let expectedNextDayPrice: Double
    let vixIndex: Double?
    let fearGreedIndex: FearGreedIndex
    let news: StockNewsData
    let chartData: ChartData?
    
    enum CodingKeys: String, CodingKey {
        case decision
        case percentage
        case reason
        case currentPrice = "current_price"
        case expectedNextDayPrice = "expected_next_day_price"
        case vixIndex = "vix_index"
        case fearGreedIndex = "fear_greed_index"
        case news
        case chartData = "chart_data"
    }
}

// MARK: - Fear and Greed Index Model
struct FearGreedIndex: Decodable {
    let value: Double
    let description: String
    let lastUpdate: String
    
    enum CodingKeys: String, CodingKey {
        case value
        case description
        case lastUpdate = "last_update"
    }
}
