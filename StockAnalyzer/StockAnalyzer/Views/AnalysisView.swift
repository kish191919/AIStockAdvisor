import SwiftUI

struct AnalysisView: View {
    let stockSymbol: String
    let language: String
    
    @StateObject private var viewModel = StockAnalysisViewModel()
    @StateObject private var yahooViewModel = YahooChartViewModel()
    @State private var showingInvalidSymbolAlert = false
    
    var body: some View {
        ScrollView {
            if viewModel.isLoading {
                LoadingView()
            } else if let error = viewModel.error {
                ErrorView(message: error)
            } else if let result = viewModel.result {
                VStack(spacing: 20) {
                    // Chart Section
                    VStack(alignment: .leading) {
                        if !stockSymbol.isEmpty {
                            YahooFinanceChartView(symbol: stockSymbol)
                                .frame(height: 400)
                        }
                    }
                    .padding()
                    .background(ColorTheme.background)
                    .cornerRadius(10)
                    .shadow(radius: 2)
                    
                    // Analysis Results
                    ResultView(result: result)
                        .slideTransition(isPresented: true)
                    
                    // News Section
                    NewsListView(news: result.news)
                        .slideTransition(isPresented: true)
                }
                .padding()
            } else {
                // Start analysis automatically
                Color.clear.onAppear {
                    Task {
                        await viewModel.analyzeStock(symbol: stockSymbol, language: language)
                        yahooViewModel.fetchChartData(symbol: stockSymbol, period: .oneDay)
                    }
                }
            }
        }
        .navigationBarTitleDisplayMode(.inline)
        .navigationTitle(stockSymbol)
        .alert("Invalid Symbol", isPresented: $showingInvalidSymbolAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(viewModel.error ?? "Invalid stock symbol")
        }
    }
}

struct AnalysisView_Previews: PreviewProvider {
    static var previews: some View {
        NavigationView {
            AnalysisView(stockSymbol: "AAPL", language: "en")
        }
    }
}
