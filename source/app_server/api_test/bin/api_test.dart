// import 'package:api_test/api_test.dart' as api_test;
import 'dart:convert';

import 'package:http/http.dart' as http;

void loadDataset() {}

Future<void> main(List<String> arguments) async {
  // print('Hello world: ${api_test.calculate()}!');
  var host = '127.0.0.1:5000';
  var url = Uri.http(host, 'loadWindows');
  var response = await http.post(url, body: {
    'granularity': 'months',
    'dataset': 'brasil',
    'pollutants': jsonEncode(['CO']),
  });

  // url = Uri.http(host, 'datasets');
  // response = await http.post(url);
  print(response.body);
}
