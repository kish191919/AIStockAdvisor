import SwiftUI
import Charts
import Combine

struct ContentView: View {
    @StateObject private var viewModel = StockAnalysisViewModel()
    @StateObject private var yahooViewModel = YahooChartViewModel()
    @StateObject private var symbolValidator = StockSymbolValidator()
    @StateObject private var recentSymbolsManager = RecentSymbolsManager()
    @StateObject private var autocomplete = StockSymbolAutocomplete()
    
    @State private var stockSymbol = ""
    @State private var selectedLanguage = "en"
    @State private var showingInvalidSymbolAlert = false
    @State private var showSuggestions = false
    @FocusState private var isTextFieldFocused: Bool
    
    let languages = [
        "af": "Afrikaans", "sq": "Shqip", "ar": "العربية",
        "hy": "Հայերեն", "az": "Azərbaycanca", "eu": "Euskara",
        "be": "Беларуская", "bg": "Български", "yue": "廣東話",
        "ca": "Català", "zh-CN": "中文（简体）", "zh-TW": "中文（繁體）",
        "hr": "Hrvatski", "cs": "Čeština", "da": "Dansk",
        "nl": "Nederlands", "en": "English", "eo": "Esperanto",
        "et": "Eesti", "tl": "Filipino", "fi": "Suomi",
        "fr": "Français", "gl": "Galego", "ka": "ქართული",
        "de": "Deutsch", "el": "Ελληνικά", "gu": "ગુજરાતી",
        "hu": "Magyar", "is": "Íslenska", "it": "Italiano",
        "ja": "日本語", "ko": "한국어", "lv": "Latviešu",
        "lt": "Lietuvių", "mk": "Македонски", "ms": "Bahasa Melayu",
        "ml": "മലയാളം", "mn": "Монгол", "no": "Norsk",
        "fa": "فارسی", "pl": "Polski", "pt": "Português",
        "ro": "Română", "ru": "Русский", "sr": "Српски",
        "sk": "Slovenčina", "sl": "Slovenščina", "es": "Español",
        "sw": "Kiswahili", "sv": "Svenska", "ta": "தமிழ்",
        "th": "ไทย", "tr": "Türkçe", "uk": "Українська",
        "vi": "Tiếng Việt", "cy": "Cymraeg", "ga": "Gaeilge",
        "ha": "Hausa", "hi": "हिन्दी", "ur": "اردو",
        "ht": "Kreyòl ayisyen", "sn": "Shona", "si": "සිංහල"
    ]
    
    private func handleSymbolChange(_ newValue: String) {
        stockSymbol = newValue.uppercased()
        showSuggestions = !newValue.isEmpty
        _ = symbolValidator.validateSymbolFormat(stockSymbol)
        autocomplete.getSuggestions(for: newValue)
    }
    
    private func handleSymbolSelection(_ symbol: String) {
        stockSymbol = symbol
        showSuggestions = false
        handleAnalysisSubmit()
    }
    
    private func handleAnalysisSubmit() {
        Task {
            isTextFieldFocused = false
            
            // 심볼 형식 검증
            guard symbolValidator.validateSymbolFormat(stockSymbol) else {
                showingInvalidSymbolAlert = true
                return
            }
            
            // 심볼 존재 여부 확인
            guard await symbolValidator.verifySymbolExists(stockSymbol) else {
                showingInvalidSymbolAlert = true
                return
            }
            
            // 분석 실행
            await viewModel.analyzeStock(symbol: stockSymbol, language: selectedLanguage)
            if viewModel.result != nil {
                viewModel.savePrediction(symbol: stockSymbol)
                recentSymbolsManager.addSymbol(stockSymbol)
            }
            yahooViewModel.fetchChartData(symbol: stockSymbol, period: .oneDay)
        }
    }
    
    var inputSection: some View {
        VStack(spacing: 15) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 4) {
                    TextField("Stock Symbol (e.g., AAPL)", text: $stockSymbol)
                        .textInputAutocapitalization(.characters)
                        .focused($isTextFieldFocused)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        #if compiler(>=5.9)
                        .onChange(of: stockSymbol) { _, newValue in
                            handleSymbolChange(newValue)
                        }
                        #else
                        .onChange(of: stockSymbol) { newValue in
                            handleSymbolChange(newValue)
                        }
                        #endif
                    
                    if !symbolValidator.isValid {
                        Text(symbolValidator.errorMessage)
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                }
                
                Picker("Language", selection: $selectedLanguage) {
                    ForEach(languages.sorted(by: { $0.value < $1.value }), id: \.key) { key, value in
                        Text(value).tag(key)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                .frame(width: 100)
            }
            
            Button(action: handleAnalysisSubmit) {
                HStack {
                    if viewModel.isLoading {
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                    } else {
                        Image(systemName: "magnifyingglass")
                        Text("Analyze Stock")
                    }
                }
                .foregroundColor(.white)
                .frame(maxWidth: .infinity)
                .frame(height: 44)
                .background(
                    stockSymbol.isEmpty || viewModel.isLoading
                    ? Color.blue.opacity(0.5)
                    : Color.blue
                )
                .cornerRadius(10)
            }
            .disabled(stockSymbol.isEmpty || viewModel.isLoading)
        }
        .padding()
    }
    
    var resultsSection: some View {
        ScrollView {
            if viewModel.isLoading {
                LoadingView()
            } else if let error = viewModel.error {
                ErrorView(message: error)
            } else if let result = viewModel.result {
                ResultsContent(stockSymbol: stockSymbol, result: result)
            }
        }
        .onTapGesture {
            isTextFieldFocused = false
            showSuggestions = false
        }
    }
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 0) {
                    inputSection
                    
                    RecentSymbolsView(manager: recentSymbolsManager) { symbol in
                        handleSymbolSelection(symbol)
                    }
                    
                    if showSuggestions && isTextFieldFocused {
                        SymbolAutocompleteView(suggestions: autocomplete.suggestions) { symbol in
                            handleSymbolSelection(symbol)
                        }
                    }
                    
                    resultsSection
                }
            }
            .navigationTitle("Stock Analyzer")
            .alert("Invalid Symbol", isPresented: $showingInvalidSymbolAlert) {
                Button("OK", role: .cancel) { }
            } message: {
                Text(symbolValidator.errorMessage)
            }
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    NavigationLink("History") {
                        StockAnalysisHistoryView()
                    }
                }
            }
        }
    }
}

// MARK: - Supporting Views
private struct LoadingView: View {
    var body: some View {
        VStack(spacing: 20) {
            ProgressView()
                .progressViewStyle(CircularProgressViewStyle())
                .scaleEffect(1.5)
            Text("Analyzing...")
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, minHeight: 200)
    }
}

private struct ResultsContent: View {
    let stockSymbol: String
    let result: StockAnalysisResponse
    
    var body: some View {
        VStack(spacing: 20) {
            VStack(alignment: .leading) {
                if !stockSymbol.isEmpty {
                    YahooFinanceChartView(symbol: stockSymbol)
                        .frame(height: 400)
                }
            }
            .padding()
            .background(Color(.systemBackground))
            .cornerRadius(10)
            .shadow(radius: 2)
            
            ResultView(result: result)
            NewsListView(news: result.news)
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
