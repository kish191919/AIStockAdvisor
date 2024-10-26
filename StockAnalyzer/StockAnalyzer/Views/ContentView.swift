import SwiftUI
import Charts
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

// Views/ContentView.swift
struct ContentView: View {
    @StateObject private var viewModel = StockAnalysisViewModel()
    @State private var stockSymbol = ""
    @State private var selectedLanguage = "en"
    
    let languages = [
        "en": "English",
        "ko": "한국어",
        "ja": "日本語",
        "zh": "中文"
    ]
    
    var body: some View {
        NavigationView {
            VStack {
                // Input Section
                VStack(spacing: 20) {
                    TextField("Stock Symbol (e.g., AAPL)", text: $stockSymbol)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .autocapitalization(.allCharacters)
                    
                    Picker("Language", selection: $selectedLanguage) {
                        ForEach(languages.sorted(by: { $0.value < $1.value }), id: \.key) { key, value in
                            Text(value).tag(key)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    
                    Button(action: {
                        viewModel.analyzeStock(symbol: stockSymbol, language: selectedLanguage)
                    }) {
                        Text("Analyze Stock")
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    .disabled(stockSymbol.isEmpty || viewModel.isLoading)
                }
                .padding()
                
                // Results Section
                ScrollView {
                    if viewModel.isLoading {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle())
                            .scaleEffect(1.5)
                    } else if let error = viewModel.error {
                        ErrorView(message: error)
                    } else if let result = viewModel.result {
                        VStack(spacing: 20) {
                            // Decision and Price Section
                            ResultView(result: result)
                            
                            // Chart Section
                            if let chartData = viewModel.chartData {
                                StockChartView(chartData: chartData)
                            }
                            
                            // News Section - news는 이미 StockAnalysisResponse 내부에 있으므로
                            // 별도의 옵셔널 체크가 필요하지 않음
                            NewsListView(news: result.news)
                        }
                        .padding()
                    }
                }
            }
            .navigationTitle("Stock Analyzer")
        }
    }
}

// PreviewProvider 추가
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
