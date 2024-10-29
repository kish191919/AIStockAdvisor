
import SwiftUI
import Charts
import Combine  // ObservableObject와 @Published를 사용하기 위해 필요

// Views/ContentView.swift

struct ContentView: View {
    @StateObject private var viewModel = StockAnalysisViewModel()
    @StateObject private var yahooViewModel = YahooChartViewModel()
    @State private var stockSymbol = ""
    @State private var selectedLanguage = "en"
    @FocusState private var isTextFieldFocused: Bool
    
    let languages = [
        "af": "Afrikaans",
        "sq": "Shqip",
        "ar": "العربية",
        "hy": "Հայերեն",
        "az": "Azərbaycanca",
        "eu": "Euskara",
        "be": "Беларуская",
        "bg": "Български",
        "yue": "廣東話",
        "ca": "Català",
        "zh-CN": "中文（简体）",
        "zh-TW": "中文（繁體）",
        "hr": "Hrvatski",
        "cs": "Čeština",
        "da": "Dansk",
        "nl": "Nederlands",
        "en": "English",
        "eo": "Esperanto",
        "et": "Eesti",
        "tl": "Filipino",
        "fi": "Suomi",
        "fr": "Français",
        "gl": "Galego",
        "ka": "ქართული",
        "de": "Deutsch",
        "el": "Ελληνικά",
        "gu": "ગુજરાતી",
        "hu": "Magyar",
        "is": "Íslenska",
        "it": "Italiano",
        "ja": "日本語",
        "ko": "한국어",
        "lv": "Latviešu",
        "lt": "Lietuvių",
        "mk": "Македонски",
        "ms": "Bahasa Melayu",
        "ml": "മലയാളം",
        "mn": "Монгол",
        "no": "Norsk",
        "fa": "فارسی",
        "pl": "Polski",
        "pt": "Português",
        "ro": "Română",
        "ru": "Русский",
        "sr": "Српски",
        "sk": "Slovenčina",
        "sl": "Slovenščina",
        "es": "Español",
        "sw": "Kiswahili",
        "sv": "Svenska",
        "ta": "தமிழ்",
        "th": "ไทย",
        "tr": "Türkçe",
        "uk": "Українська",
        "vi": "Tiếng Việt",
        "cy": "Cymraeg",
        "ga": "Gaeilge",
        "ha": "Hausa",
        "hi": "हिन्दी",
        "ur": "اردو",
        "ht": "Kreyòl ayisyen",
        "sn": "Shona",
        "si": "සිංහල"
    ]
    
    var body: some View {
            NavigationView {
                ScrollView {
                    VStack(spacing: 0) {
                        // Input Section
                        VStack(spacing: 15) {
                            HStack(spacing: 10) {
                                TextField("Stock Symbol (e.g., AAPL)", text: $stockSymbol)
                                    .autocapitalization(.allCharacters)
                                    .focused($isTextFieldFocused)
                                    .frame(height: 44)
                                    .padding(.horizontal, 10)
                                    .background(
                                        RoundedRectangle(cornerRadius: 10)
                                            .stroke(Color.gray.opacity(0.3))
                                            .background(Color(.systemBackground))
                                    )
                                
                                Picker("Language", selection: $selectedLanguage) {
                                    ForEach(languages.sorted(by: { $0.value < $1.value }), id: \.key) { key, value in
                                        Text(value).tag(key)
                                    }
                                }
                                .pickerStyle(MenuPickerStyle())
                                .frame(width: 100)
                            }
                            
                            Button {
                                isTextFieldFocused = false
                                Task {
                                    await viewModel.analyzeStock(symbol: stockSymbol, language: selectedLanguage)
                                    if viewModel.result != nil {
                                        viewModel.savePrediction(symbol: stockSymbol)
                                    }
                                    yahooViewModel.fetchChartData(symbol: stockSymbol, period: .oneDay)
                                }
                            } label: {
                                Text("Analyze Stock")
                                    .foregroundColor(.white)
                                    .frame(maxWidth: .infinity)
                                    .frame(height: 44)
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
                                    .padding()
                            } else if let error = viewModel.error {
                                VStack(spacing: 10) {
                                    Image(systemName: "exclamationmark.triangle")
                                        .font(.system(size: 40))
                                        .foregroundColor(.orange)
                                    
                                    Text(error)
                                        .multilineTextAlignment(.center)
                                        .foregroundColor(.secondary)
                                    
                                    Button("Try Again") {
                                        Task {
                                            await viewModel.analyzeStock(symbol: stockSymbol, language: selectedLanguage)
                                            yahooViewModel.fetchChartData(symbol: stockSymbol, period: .oneDay)
                                        }
                                    }
                                    .padding()
                                    .background(Color.blue)
                                    .foregroundColor(.white)
                                    .cornerRadius(8)
                                }
                                .padding()
                            } else if let result = viewModel.result {
                                VStack(spacing: 20) {
                                    // Yahoo Finance Chart Section
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
                                    
                                    // Analysis Results
                                    ResultView(result: result)
                                    
                                    // News Section
                                    NewsListView(news: result.news)
                                }
                                .padding()
                            }
                        }
                        .onTapGesture {
                            isTextFieldFocused = false
                        }
                    }
                }
                .navigationTitle("Stock Analyzer")
                .onTapGesture {
                    isTextFieldFocused = false
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

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
