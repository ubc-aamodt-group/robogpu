#ifndef CSV_READER_H
#define CSV_READER_H

#include <istream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

enum class CSVState {
    UnquotedField,
    QuotedField,
    QuotedQuote
};

std::vector<std::string> readCSVRow(const std::string &row) {
    CSVState state = CSVState::UnquotedField;
    std::vector<std::string> fields {""};
    size_t i = 0; // index of the current field
    for (char c : row) {
        switch (state) {
            case CSVState::UnquotedField:
                switch (c) {
                    case ',': // end of field
                              fields.push_back(""); i++;
                              break;
                    case '"': state = CSVState::QuotedField;
                              break;
                    default:  fields[i].push_back(c);
                              break; }
                break;
            case CSVState::QuotedField:
                switch (c) {
                    case '"': state = CSVState::QuotedQuote;
                              break;
                    default:  fields[i].push_back(c);
                              break; }
                break;
            case CSVState::QuotedQuote:
                switch (c) {
                    case ',': // , after closing quote
                              fields.push_back(""); i++;
                              state = CSVState::UnquotedField;
                              break;
                    case '"': // "" -> "
                              fields[i].push_back('"');
                              state = CSVState::QuotedField;
                              break;
                    default:  // end of quote
                              state = CSVState::UnquotedField;
                              break; }
                break;
        }
    }
    return fields;
}

/// Read CSV file, Excel dialect. Accept "quoted fields ""with quotes"""
std::vector<std::vector<std::string>> readCSV(std::istream &in) {
    std::vector<std::vector<std::string>> table;
    std::string row;
    while (!in.eof()) {
        std::getline(in, row);
        if (in.bad() || in.fail()) {
            break;
        }
        auto fields = readCSVRow(row);
        table.push_back(fields);
    }
    return table;
}

std::vector<std::string> split(std::string input, char delimiter) {
    std::stringstream ss(input);
    std::vector<std::string> result;
    while( ss.good() )
    {
        std::string substr;
        std::getline( ss, substr, delimiter );
        result.push_back( substr );
    }
    return result;
}

std::vector<struct geometry::OBB<double>> load_obbs(std::string filename) {
    std::vector<struct geometry::OBB<double>> links;

    std::ifstream file;
    file.open(filename);

    std::vector<std::vector<std::string>> csv_data;
    csv_data = readCSV(file);

    // csv format: pos,joint,obb_center_x,obb_center_y,obb_center_z,obb_axis_x1,obb_axis_y1,obb_axis_z1,obb_axis_x2,obb_axis_y2,obb_axis_z2,obb_axis_x3,obb_axis_y3,obb_axis_z3,obb_half_length_x,obb_half_length_y,obb_half_length_z
    for (int i = 1; i < csv_data.size(); i++) { // skip first row which are the headers
        struct geometry::OBB<double> obb;
        obb.c[0] = std::stod(csv_data[i][2]);
        obb.c[1] = std::stod(csv_data[i][3]);
        obb.c[2] = std::stod(csv_data[i][4]);

        obb.e[0] = std::stod(csv_data[i][14]);
        obb.e[1] = std::stod(csv_data[i][15]);
        obb.e[2] = std::stod(csv_data[i][16]);

        obb.u[0][0] = std::stod(csv_data[i][5]);
        obb.u[0][1] = std::stod(csv_data[i][8]);
        obb.u[0][2] = std::stod(csv_data[i][11]);

        obb.u[1][0] = std::stod(csv_data[i][6]);
        obb.u[1][1] = std::stod(csv_data[i][9]);
        obb.u[1][2] = std::stod(csv_data[i][12]);

        obb.u[2][0] = std::stod(csv_data[i][7]);
        obb.u[2][1] = std::stod(csv_data[i][10]);
        obb.u[2][2] = std::stod(csv_data[i][13]);
        links.push_back(obb);
    }
    return links;
}

#endif // CSV_READER_H