import SwiftUI
import Charts
import Combine

struct ContentView: View {
    // MARK: - Properties
    @StateObject private var viewModel = StockAnalysisViewModel()
    @StateObject private var yahooViewModel = YahooChartViewModel()
    @StateObject private var symbolValidator = StockSymbolValidator()
    @StateObject private var recentSymbolsManager = RecentSymbolsManager()
    @StateObject private var autocomplete = StockSymbolAutocomplete()
    
    @State private var stockSymbol = ""
    @State private var selectedLanguage = "en"
    @State private var showingInvalidSymbolAlert = false
    @State private var showSuggestions = false
    @State private var isViewAppeared = false
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
    
    // MARK: - Methods
    private func handleSymbolChange(_ newValue: String) {
        withAnimation {
            stockSymbol = newValue.uppercased()
            showSuggestions = !newValue.isEmpty
            _ = symbolValidator.validateSymbolFormat(stockSymbol)
            autocomplete.getSuggestions(for: newValue)
        }
    }
    
    private func handleSymbolSelection(_ symbol: String) {
        withAnimation {
            stockSymbol = symbol
            showSuggestions = false
            handleAnalysisSubmit()
        }
    }
    
    private func handleAnalysisSubmit() {
        Task {
            isTextFieldFocused = false
            
            guard symbolValidator.validateSymbolFormat(stockSymbol) else {
                showingInvalidSymbolAlert = true
                return
            }
            
            guard await symbolValidator.verifySymbolExists(stockSymbol) else {
                showingInvalidSymbolAlert = true
                return
            }
            
            await viewModel.analyzeStock(symbol: stockSymbol, language: selectedLanguage)
            if viewModel.result != nil {
                withAnimation {
                    viewModel.savePrediction(symbol: stockSymbol)
                    recentSymbolsManager.addSymbol(stockSymbol)
                }
                yahooViewModel.fetchChartData(symbol: stockSymbol, period: .oneDay)
            }
        }
    }
    
    // MARK: - View Components
    var searchBar: some View {
        HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                TextField("Stock Symbol (e.g., AAPL)", text: $stockSymbol)
                    .textInputAutocapitalization(.characters)
                    .focused($isTextFieldFocused)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .foregroundColor(ColorTheme.text)
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
                        .foregroundColor(ColorTheme.error)
                        .slideTransition(isPresented: !symbolValidator.isValid)
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
    }
    
    var searchButton: some View {
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
                ? ColorTheme.accent.opacity(0.5)
                : ColorTheme.accent
            )
            .cornerRadius(10)
        }
        .disabled(stockSymbol.isEmpty || viewModel.isLoading)
        .slideTransition(isPresented: !stockSymbol.isEmpty)
    }
    
    var searchSection: some View {
        VStack(spacing: 15) {
            searchBar
            searchButton
        }
        .padding()
        .background(ColorTheme.background)
        .slideTransition(isPresented: isViewAppeared)
    }
    
    var recentSearchesSection: some View {
        RecentSymbolsView(manager: recentSymbolsManager) { symbol in
            handleSymbolSelection(symbol)
        }
        .slideTransition(isPresented: isViewAppeared)
    }
    
    var suggestionsSection: some View {
        Group {
            if showSuggestions && isTextFieldFocused {
                SymbolAutocompleteView(suggestions: autocomplete.suggestions) { symbol in
                    handleSymbolSelection(symbol)
                }
                .slideTransition(isPresented: showSuggestions)
            }
        }
    }
    
    var resultsSection: some View {
        ScrollView {
            if viewModel.isLoading {
                LoadingView()
                    .slideTransition(isPresented: viewModel.isLoading)
            } else if let error = viewModel.error {
                ErrorView(message: error)
                    .slideTransition(isPresented: true)
            } else if let result = viewModel.result {
                ResultsContent(
                    stockSymbol: stockSymbol,
                    result: result,
                    yahooViewModel: yahooViewModel
                )
                .slideTransition(isPresented: true)
            }
        }
        .onTapGesture {
            isTextFieldFocused = false
            showSuggestions = false
        }
    }
    
    // MARK: - Body
    var body: some View {
        NavigationView {
            ZStack {
                ColorTheme.secondaryBackground
                    .ignoresSafeArea()
                
                ScrollView {
                    VStack(spacing: 0) {
                        searchSection
                        recentSearchesSection
                        suggestionsSection
                        resultsSection
                    }
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
        .onAppear {
            withAnimation(.easeOut(duration: 0.5)) {
                isViewAppeared = true
            }
        }
    }
}

// MARK: - ResultsContent View
struct ResultsContent: View {
    let stockSymbol: String
    let result: StockAnalysisResponse
    @ObservedObject var yahooViewModel: YahooChartViewModel
    @State private var selectedPoint: YahooChartDataPoint?
    
    var body: some View {
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
    }
}

// MARK: - Preview Provider
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
            .preferredColorScheme(.light)
        
        ContentView()
            .preferredColorScheme(.dark)
    }
}
