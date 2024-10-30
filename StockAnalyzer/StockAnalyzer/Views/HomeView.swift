import SwiftUI

struct HomeView: View {
    @State private var stockSymbol = ""
    @State private var selectedLanguage = "en"
    @State private var showAnalysis = false
    @StateObject private var symbolValidator = StockSymbolValidator()
    @StateObject private var recentSymbolsManager = RecentSymbolsManager()
    @StateObject private var autocomplete = StockSymbolAutocomplete()
    
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
    
    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Search Section
                    VStack(spacing: 15) {
                        HStack(spacing: 10) {
                            // Symbol Input Field
                            TextField("Stock Symbol (e.g., AAPL)", text: $stockSymbol)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .textInputAutocapitalization(.characters)
                                .onChange(of: stockSymbol) { newValue in
                                    stockSymbol = newValue.uppercased()
                                    _ = symbolValidator.validateSymbolFormat(stockSymbol)
                                    autocomplete.getSuggestions(for: newValue)
                                }
                            
                            // Language Picker
                            Picker("Language", selection: $selectedLanguage) {
                                ForEach(languages.sorted(by: { $0.value < $1.value }), id: \.key) { key, value in
                                    Text(value).tag(key)
                                }
                            }
                            .pickerStyle(MenuPickerStyle())
                            .frame(width: 100)
                        }
                        
                        NavigationLink(
                            destination: AnalysisView(
                                stockSymbol: stockSymbol,
                                language: selectedLanguage
                            ),
                            isActive: $showAnalysis
                        ) {
                            Button(action: {
                                if !stockSymbol.isEmpty {
                                    showAnalysis = true
                                }
                            }) {
                                HStack {
                                    Image(systemName: "magnifyingglass")
                                    Text("Analyze Stock")
                                }
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .frame(height: 44)
                                .background(stockSymbol.isEmpty ? Color.blue.opacity(0.5) : Color.blue)
                                .cornerRadius(10)
                            }
                        }
                        .disabled(stockSymbol.isEmpty)
                    }
                    .padding()
                    
                    // Recent Searches
                    RecentSymbolsView(manager: recentSymbolsManager) { symbol in
                        stockSymbol = symbol
                        showAnalysis = true
                    }
                    
                    // Autocomplete Suggestions
                    if !stockSymbol.isEmpty {
                        SymbolAutocompleteView(suggestions: autocomplete.suggestions) { symbol in
                            stockSymbol = symbol
                            showAnalysis = true
                        }
                    }
                    
                    Spacer()
                }
            }
            .navigationTitle("Stock Analyzer")
            .background(ColorTheme.secondaryBackground)
        }
    }
}
